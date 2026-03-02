#!/usr/bin/env python3
"""
AI Shaman â€” Ollama GPU Guardian
a'shamon: "Guardian" in the Old Tongue

Async reverse proxy that sits between consumers and Ollama instances,
enforcing power budgets, temperature circuit breakers, and request
queuing to prevent simultaneous GPU power spikes that cause hard reboots.

Zero-change deployment: consumers hit the same ports as before.
"""

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
from aiohttp import web

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GpuConfig:
    id: int
    name: str
    listen_port: int
    backend_port: int
    max_concurrent: int = 1
    temp_pause_c: float = 82.0
    temp_resume_c: float = 75.0
    power_limit_w: float = 200.0
    allowed_models: list[str] = field(default_factory=list)  # e.g. ["qwen3:*", "qwen2.5:*"]


@dataclass
class Config:
    gpus: list[GpuConfig] = field(default_factory=list)
    combined_power_threshold_w: float = 350.0
    monitor_interval_s: float = 3.0
    queue_timeout_s: float = 300.0
    log_file: str = "/home/om/ollama-guardian/guardian.log"
    management_port: int = 11450
    clean_output: bool = True
    request_no_think_hint: bool = True

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        gpu_dicts = raw.get("gpus", [])
        gpus = []
        for g in gpu_dicts:
            allowed = g.pop("allowed_models", [])
            gc = GpuConfig(**g)
            gc.allowed_models = allowed
            gpus.append(gc)
        return cls(
            gpus=gpus,
            combined_power_threshold_w=raw.get("combined_power_threshold_w", 350.0),
            monitor_interval_s=raw.get("monitor_interval_s", 3.0),
            queue_timeout_s=raw.get("queue_timeout_s", 300.0),
            log_file=raw.get("log_file", cls.log_file),
            management_port=raw.get("management_port", 11450),
            clean_output=raw.get("clean_output", True),
            request_no_think_hint=raw.get("request_no_think_hint", True),
        )


# ---------------------------------------------------------------------------
# GPU Monitor
# ---------------------------------------------------------------------------

@dataclass
class GpuStats:
    gpu_id: int
    temp_c: float = 0.0
    power_w: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_free_mb: float = 0.0
    timestamp: float = 0.0


class GpuMonitor:
    """Polls nvidia-smi and caches per-GPU stats."""

    def __init__(self, gpu_ids: list[int], interval: float):
        self.gpu_ids = gpu_ids
        self.interval = interval
        self.stats: dict[int, GpuStats] = {gid: GpuStats(gpu_id=gid) for gid in gpu_ids}
        self._running = False
        self.power_history: deque[tuple[float, float]] = deque(maxlen=1200)  # ~1hr at 3s

    async def start(self):
        self._running = True
        asyncio.create_task(self._poll_loop())

    async def stop(self):
        self._running = False

    async def _poll_loop(self):
        while self._running:
            try:
                await self._poll_once()
            except Exception as e:
                logging.warning(f"nvidia-smi poll error: {e}")
            await asyncio.sleep(self.interval)

    async def _poll_once(self):
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except asyncio.TimeoutError:
            proc.kill()
            logging.error("nvidia-smi timed out (>10s) â€” GPU may be hung")
            return
        now = time.time()
        total_power = 0.0
        for line in stdout.decode().strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gid = int(parts[0])
            if gid in self.stats:
                s = self.stats[gid]
                s.temp_c = float(parts[1])
                s.power_w = float(parts[2])
                s.vram_used_mb = float(parts[3])
                s.vram_total_mb = float(parts[4])
                s.vram_free_mb = float(parts[5])
                s.timestamp = now
                total_power += s.power_w
        self.power_history.append((now, total_power))

    @property
    def combined_power(self) -> float:
        return sum(s.power_w for s in self.stats.values())


# ---------------------------------------------------------------------------
# Per-GPU Queue & Circuit Breaker
# ---------------------------------------------------------------------------

class GpuGate:
    """Controls access to a single GPU: semaphore + circuit breaker state."""

    def __init__(self, gpu_cfg: GpuConfig):
        self.cfg = gpu_cfg
        self.semaphore = asyncio.Semaphore(gpu_cfg.max_concurrent)
        self.paused = False           # manual pause
        self.temp_tripped = False     # auto-pause from temperature
        self.power_tripped = False    # auto-pause from combined power
        self.queue_depth = 0
        self.total_requests = 0
        self.total_errors = 0
        self.latency_samples: deque[float] = deque(maxlen=500)
        self._resume_event = asyncio.Event()
        self._resume_event.set()      # starts unblocked

    def trip_temp(self):
        if not self.temp_tripped:
            self.temp_tripped = True
            self._resume_event.clear()
            logging.warning(f"[{self.cfg.name}] TEMP CIRCUIT BREAKER TRIPPED â€” GPU {self.cfg.id} too hot")

    def clear_temp(self):
        if self.temp_tripped:
            self.temp_tripped = False
            if not self.paused and not self.power_tripped:
                self._resume_event.set()
            logging.info(f"[{self.cfg.name}] Temp circuit breaker cleared")

    def trip_power(self):
        if not self.power_tripped:
            self.power_tripped = True
            self._resume_event.clear()
            logging.warning(f"[{self.cfg.name}] POWER BUDGET EXCEEDED â€” queuing requests")

    def clear_power(self):
        if self.power_tripped:
            self.power_tripped = False
            if not self.paused and not self.temp_tripped:
                self._resume_event.set()
            logging.info(f"[{self.cfg.name}] Power budget cleared")

    def manual_pause(self):
        self.paused = True
        self._resume_event.clear()
        logging.info(f"[{self.cfg.name}] Manually paused")

    def manual_resume(self):
        self.paused = False
        if not self.temp_tripped and not self.power_tripped:
            self._resume_event.set()
        logging.info(f"[{self.cfg.name}] Manually resumed")

    @property
    def is_blocked(self) -> bool:
        return self.paused or self.temp_tripped or self.power_tripped

    async def acquire(self, timeout: float):
        """Wait for circuit breaker + semaphore. Raises TimeoutError."""
        self.queue_depth += 1
        try:
            # Wait for circuit breaker to clear
            deadline = asyncio.get_event_loop().time() + timeout
            while self.is_blocked:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"Queue timeout waiting for GPU {self.cfg.name}")
                try:
                    await asyncio.wait_for(self._resume_event.wait(), timeout=min(remaining, 1.0))
                except asyncio.TimeoutError:
                    continue  # re-check is_blocked

            # Acquire semaphore
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Queue timeout waiting for GPU {self.cfg.name}")
            await asyncio.wait_for(self.semaphore.acquire(), timeout=remaining)
        finally:
            self.queue_depth -= 1

    def release(self):
        self.semaphore.release()


# ---------------------------------------------------------------------------
# Request Logger
# ---------------------------------------------------------------------------

class RequestLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self._fh = None

    async def start(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._fh = open(self.log_file, "a", buffering=1)

    async def stop(self):
        if self._fh:
            self._fh.close()

    def log(self, gpu_name: str, method: str, path: str, status: int,
            latency_ms: float, power_w: float, model: str = ""):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        entry = (f"{ts} | gpu={gpu_name} | {method} {path} | model={model} "
                 f"| status={status} | latency={latency_ms:.0f}ms | power={power_w:.1f}W\n")
        if self._fh:
            self._fh.write(entry)


# ---------------------------------------------------------------------------
# Proxy App (one per GPU)
# ---------------------------------------------------------------------------

# Inference endpoints that go through the queue
QUEUED_PATHS = {
    "/v1/chat/completions",
    "/api/generate",
    "/api/chat",
    "/api/embed",
    "/api/embeddings",
}


# Dangerous admin endpoints â€” blocked entirely at the proxy
# These can load/unload/delete models, thrashing VRAM and bypassing affinity.
BLOCKED_PATHS = {
    "/api/pull",
    "/api/push",
    "/api/delete",
    "/api/copy",
    "/api/create",
}


def _is_queued(path: str) -> bool:
    return path in QUEUED_PATHS


def _model_matches(model: str, pattern: str) -> bool:
    """Check if a model name matches an allowed pattern.

    Supports:
      - Exact match: "qwen3:14b" matches "qwen3:14b"
      - Wildcard suffix: "qwen3:*" matches "qwen3:14b", "qwen3:14b-nothinker-16k", etc.
      - Base name wildcard: "qwen3" matches "qwen3:anything" (bare name = all tags)
    """
    if pattern == model:
        return True
    if pattern.endswith(":*"):
        prefix = pattern[:-2]  # "qwen3"
        return model == prefix or model.startswith(prefix + ":")
    # Bare name without colon = match all tags of that model
    if ":" not in pattern:
        return model == pattern or model.startswith(pattern + ":")
    return False


def _check_model_allowed(model: str, gpu_cfg: GpuConfig) -> bool:
    """Return True if the model is allowed on this GPU (or no restrictions configured)."""
    if not gpu_cfg.allowed_models:
        return True  # no restrictions = allow all
    return any(_model_matches(model, pat) for pat in gpu_cfg.allowed_models)


async def _extract_model(request: web.Request) -> str:
    """Try to extract model name from request body without consuming it."""
    try:
        body = await request.read()
        data = json.loads(body)
        return data.get("model", "")
    except Exception:
        return ""


def _dedupe_repeated_sentences(text: str) -> str:
    """Collapse pathological repeated sentence tails."""
    s = (text or "").strip()
    if not s:
        return s
    # Split on sentence boundaries and drop immediate repeats.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
    if len(parts) <= 1:
        return s
    out = [parts[0]]
    repeat_count = 0
    for p in parts[1:]:
        if p == out[-1]:
            repeat_count += 1
            # Keep at most one duplicate.
            if repeat_count > 1:
                continue
        else:
            repeat_count = 0
        out.append(p)
    return " ".join(out)


def _sanitize_json_payload(payload: dict | list, path: str) -> dict | list:
    """Remove noisy reasoning fields and tame repetition in final content."""
    if not isinstance(payload, dict):
        return payload

    # OpenAI-compatible /v1/chat/completions
    if path == "/v1/chat/completions":
        for choice in payload.get("choices", []) if isinstance(payload.get("choices"), list) else []:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message")
            if not isinstance(msg, dict):
                continue
            msg.pop("reasoning", None)
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _dedupe_repeated_sentences(content)

    # Ollama /api/chat non-stream JSON shape
    if path == "/api/chat":
        msg = payload.get("message")
        if isinstance(msg, dict):
            msg.pop("thinking", None)
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _dedupe_repeated_sentences(content)

    # Ollama /api/generate non-stream JSON shape
    if path == "/api/generate":
        if isinstance(payload.get("response"), str):
            payload["response"] = _dedupe_repeated_sentences(payload["response"])
        payload.pop("thinking", None)

    return payload


def _inject_no_think(body: bytes, path: str) -> bytes:
    """Best-effort request hint to suppress reasoning/thinking output."""
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return body
    if not isinstance(data, dict):
        return body

    if path in {"/api/chat", "/api/generate"}:
        # Ollama-native endpoints accept this hint on recent versions.
        data.setdefault("think", False)

    if path == "/v1/chat/completions":
        msgs = data.get("messages")
        if isinstance(msgs, list):
            msgs.insert(0, {
                "role": "system",
                "content": "Return final answer only. Do not include reasoning.",
            })
            data["messages"] = msgs

    try:
        return json.dumps(data).encode("utf-8")
    except Exception:
        return body


async def _proxy_request(request: web.Request, backend_port: int) -> web.StreamResponse:
    """Proxy a request to the backend Ollama and stream the response back."""
    backend_url = f"http://127.0.0.1:{backend_port}{request.path_qs}"

    # Read the body once
    body = await request.read()
    if request.method == "POST":
        app_cfg = request.app.get("guardian_config")
        if app_cfg and app_cfg.request_no_think_hint and _is_queued(request.path):
            body = _inject_no_think(body, request.path)

    # Build headers, filtering hop-by-hop
    headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl not in ("host", "transfer-encoding", "connection", "content-length"):
            headers[k] = v

    timeout = aiohttp.ClientTimeout(total=600, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                data=body,
            ) as backend_resp:
                # Check if response is streaming
                ct = backend_resp.headers.get("Content-Type", "")
                is_stream = ("text/event-stream" in ct or
                             "application/x-ndjson" in ct or
                             "ndjson" in ct)

                # For streaming, use StreamResponse
                if is_stream:
                    resp = web.StreamResponse(
                        status=backend_resp.status,
                        headers={k: v for k, v in backend_resp.headers.items()
                                 if k.lower() not in ("transfer-encoding", "connection")},
                    )
                    await resp.prepare(request)
                    async for chunk in backend_resp.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                    return resp
                else:
                    # Non-streaming: read full body
                    resp_body = await backend_resp.read()
                    app_cfg = request.app.get("guardian_config")
                    if app_cfg and app_cfg.clean_output:
                        ct = backend_resp.headers.get("Content-Type", "")
                        if "application/json" in ct:
                            try:
                                parsed = json.loads(resp_body.decode("utf-8"))
                                cleaned = _sanitize_json_payload(parsed, request.path)
                                resp_body = json.dumps(cleaned).encode("utf-8")
                            except Exception:
                                pass
                    resp = web.Response(
                        status=backend_resp.status,
                        body=resp_body,
                        headers={k: v for k, v in backend_resp.headers.items()
                                 if k.lower() not in ("transfer-encoding", "connection",
                                                       "content-length", "content-encoding")},
                    )
                    return resp
        except aiohttp.ClientError as e:
            logging.error(f"Backend connection error (port {backend_port}): {e}")
            return web.json_response(
                {"error": f"Backend Ollama unavailable: {e}"},
                status=502,
            )


def _build_routing_table(config: Config) -> list[dict]:
    """Build a human-readable routing table for error messages."""
    return [
        {
            "gpu": gcfg.name,
            "port": gcfg.listen_port,
            "allowed_models": gcfg.allowed_models or ["(any â€” no restriction)"],
        }
        for gcfg in config.gpus
    ]


def make_gpu_app(gate: GpuGate, monitor: GpuMonitor, config: Config,
                 req_logger: RequestLogger) -> web.Application:
    """Create an aiohttp app for one GPU's proxy."""

    routing_table = _build_routing_table(config)

    async def handle_request(request: web.Request) -> web.StreamResponse:
        path = request.path
        method = request.method
        gpu_stats = monitor.stats.get(gate.cfg.id)

        # â”€â”€ BLOCK DANGEROUS ADMIN ENDPOINTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if path in BLOCKED_PATHS:
            logging.warning(
                f"[{gate.cfg.name}] BLOCKED admin endpoint: {method} {path}"
            )
            req_logger.log(gate.cfg.name, method, path, 403, 0.0,
                           gpu_stats.power_w if gpu_stats else 0.0, "")
            return web.json_response({
                "error": "ENDPOINT BLOCKED BY AI SHAMAN",
                "path": path,
                "guardian": "ai-shaman",
                "message": (
                    f"The endpoint {path} is blocked by AI Shaman to prevent "
                    f"model thrashing and VRAM disruption. Use the management "
                    f"API on port {config.management_port} or access Ollama "
                    f"directly on the internal port if you need admin operations."
                ),
            }, status=403)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        # Extract model for logging and affinity check
        model = ""
        if method == "POST" and _is_queued(path):
            model = await _extract_model(request)

        # â”€â”€ MODEL AFFINITY CHECK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Fail CLOSED: if allowed_models is configured, model MUST be present and match.
        # Empty/missing model field is rejected â€” no sneaking past affinity.
        if method == "POST" and _is_queued(path) and gate.cfg.allowed_models:
            if not model:
                gate.total_requests += 1
                gate.total_errors += 1
                rejection = {
                    "error": "MODEL REJECTED BY AI SHAMAN",
                    "rejected_model": "(empty or missing)",
                    "rejected_on": {
                        "gpu_id": gate.cfg.id,
                        "gpu_name": gate.cfg.name,
                        "port": gate.cfg.listen_port,
                        "allowed_models": gate.cfg.allowed_models,
                    },
                    "hint": "Request body must include a 'model' field matching one of the allowed models.",
                    "routing_table": routing_table,
                    "guardian": "ai-shaman",
                    "message": (
                        f"REQUEST BLOCKED: No 'model' field in request body. "
                        f"GPU {gate.cfg.id} ({gate.cfg.name}) requires one of: "
                        f"{gate.cfg.allowed_models}. AI Shaman does not allow "
                        f"anonymous model requests when affinity is configured."
                    ),
                }
                logging.warning(
                    f"[{gate.cfg.name}] MODEL REJECTED: empty/missing model field â€” returning 403"
                )
                req_logger.log(gate.cfg.name, method, path, 403, 0.0,
                               gpu_stats.power_w if gpu_stats else 0.0, "")
                return web.json_response(rejection, status=403)

            if not _check_model_allowed(model, gate.cfg):
                gate.total_requests += 1
                gate.total_errors += 1
                # Find which GPU SHOULD handle this model
                correct_gpu = None
                for gcfg in config.gpus:
                    if gcfg.id != gate.cfg.id and _check_model_allowed(model, gcfg):
                        correct_gpu = gcfg
                        break

                hint = (
                    f"Send '{model}' requests to port {correct_gpu.listen_port} "
                    f"({correct_gpu.name}) instead."
                    if correct_gpu else
                    f"Model '{model}' is not configured on ANY GPU. Check your guardian_config.json."
                )

                rejection = {
                    "error": "MODEL REJECTED BY AI SHAMAN",
                    "rejected_model": model,
                    "rejected_on": {
                        "gpu_id": gate.cfg.id,
                        "gpu_name": gate.cfg.name,
                        "port": gate.cfg.listen_port,
                        "allowed_models": gate.cfg.allowed_models,
                    },
                    "hint": hint,
                    "routing_table": routing_table,
                    "guardian": "ai-shaman",
                    "message": (
                        f"REQUEST BLOCKED: Model '{model}' is NOT allowed on "
                        f"GPU {gate.cfg.id} ({gate.cfg.name}, port {gate.cfg.listen_port}). "
                        f"This GPU only serves: {gate.cfg.allowed_models}. {hint}"
                    ),
                }

                logging.warning(
                    f"[{gate.cfg.name}] MODEL REJECTED: '{model}' not in "
                    f"{gate.cfg.allowed_models} â€” returning 403"
                )
                req_logger.log(gate.cfg.name, method, path, 403, 0.0,
                               gpu_stats.power_w if gpu_stats else 0.0, model)
                return web.json_response(rejection, status=403)
        # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

        start_time = time.time()

        if method == "POST" and _is_queued(path):
            # Queued inference request
            gate.total_requests += 1
            try:
                await gate.acquire(config.queue_timeout_s)
            except TimeoutError as e:
                gate.total_errors += 1
                logging.warning(f"[{gate.cfg.name}] Request timed out in queue: {e}")
                return web.json_response(
                    {"error": str(e), "guardian": "ai-shaman"},
                    status=503,
                )

            try:
                resp = await _proxy_request(request, gate.cfg.backend_port)
                latency = (time.time() - start_time) * 1000
                gate.latency_samples.append(latency)
                power = gpu_stats.power_w if gpu_stats else 0.0
                status = resp.status if hasattr(resp, 'status') else 200
                req_logger.log(gate.cfg.name, method, path, status,
                               latency, power, model)
                if status >= 400:
                    gate.total_errors += 1
                return resp
            finally:
                gate.release()
        else:
            # Pass-through (non-inference endpoints)
            resp = await _proxy_request(request, gate.cfg.backend_port)
            latency = (time.time() - start_time) * 1000
            power = gpu_stats.power_w if gpu_stats else 0.0
            status = resp.status if hasattr(resp, 'status') else 200
            req_logger.log(gate.cfg.name, method, path, status,
                           latency, power, model)
            return resp

    app = web.Application()
    app["guardian_config"] = config
    app.router.add_route("*", "/{path_info:.*}", handle_request)
    return app


# ---------------------------------------------------------------------------
# Management API (port 11450)
# ---------------------------------------------------------------------------

def make_management_app(gates: dict[int, GpuGate], monitor: GpuMonitor,
                        config: Config) -> web.Application:
    """Management endpoints for AI Shaman status and control."""

    async def status_handler(request: web.Request) -> web.Response:
        gpu_info = []
        for gcfg in config.gpus:
            gate = gates[gcfg.id]
            stats = monitor.stats.get(gcfg.id, GpuStats(gpu_id=gcfg.id))
            gpu_info.append({
                "id": gcfg.id,
                "name": gcfg.name,
                "listen_port": gcfg.listen_port,
                "backend_port": gcfg.backend_port,
                "allowed_models": gcfg.allowed_models or ["(any)"],
                "temp_c": stats.temp_c,
                "power_w": stats.power_w,
                "vram_used_mb": stats.vram_used_mb,
                "vram_total_mb": stats.vram_total_mb,
                "vram_free_mb": stats.vram_free_mb,
                "queue_depth": gate.queue_depth,
                "total_requests": gate.total_requests,
                "total_errors": gate.total_errors,
                "is_blocked": gate.is_blocked,
                "temp_tripped": gate.temp_tripped,
                "power_tripped": gate.power_tripped,
                "paused": gate.paused,
            })
        return web.json_response({
            "guardian": "AI Shaman",
            "version": "1.0.0",
            "combined_power_w": monitor.combined_power,
            "combined_power_threshold_w": config.combined_power_threshold_w,
            "gpus": gpu_info,
        })

    async def metrics_handler(request: web.Request) -> web.Response:
        gpu_metrics = []
        for gcfg in config.gpus:
            gate = gates[gcfg.id]
            samples = list(gate.latency_samples)
            avg_lat = sum(samples) / len(samples) if samples else 0
            p95 = sorted(samples)[int(len(samples) * 0.95)] if len(samples) > 1 else avg_lat
            gpu_metrics.append({
                "id": gcfg.id,
                "name": gcfg.name,
                "total_requests": gate.total_requests,
                "total_errors": gate.total_errors,
                "avg_latency_ms": round(avg_lat, 1),
                "p95_latency_ms": round(p95, 1),
                "samples": len(samples),
            })
        # Power history (last 100 points)
        power_hist = [{"t": t, "w": w} for t, w in list(monitor.power_history)[-100:]]
        return web.json_response({
            "gpus": gpu_metrics,
            "power_history": power_hist,
        })

    async def pause_handler(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            data = {}
        gpu_id = data.get("gpu_id")
        if gpu_id is None:
            # Pause all
            for gate in gates.values():
                gate.manual_pause()
            return web.json_response({"status": "all GPUs paused"})
        if gpu_id not in gates:
            return web.json_response({"error": f"Unknown GPU {gpu_id}"}, status=404)
        gates[gpu_id].manual_pause()
        return web.json_response({"status": f"GPU {gpu_id} paused"})

    async def resume_handler(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            data = {}
        gpu_id = data.get("gpu_id")
        if gpu_id is None:
            for gate in gates.values():
                gate.manual_resume()
            return web.json_response({"status": "all GPUs resumed"})
        if gpu_id not in gates:
            return web.json_response({"error": f"Unknown GPU {gpu_id}"}, status=404)
        gates[gpu_id].manual_resume()
        return web.json_response({"status": f"GPU {gpu_id} resumed"})

    app = web.Application()
    app.router.add_get("/guardian/status", status_handler)
    app.router.add_get("/guardian/metrics", metrics_handler)
    app.router.add_post("/guardian/pause", pause_handler)
    app.router.add_post("/guardian/resume", resume_handler)
    return app


# ---------------------------------------------------------------------------
# Circuit Breaker Loop
# ---------------------------------------------------------------------------

async def circuit_breaker_loop(gates: dict[int, GpuGate], monitor: GpuMonitor,
                                config: Config):
    """Continuously check GPU temps and power, trip/clear circuit breakers."""
    while True:
        await asyncio.sleep(config.monitor_interval_s)

        combined = monitor.combined_power

        for gcfg in config.gpus:
            gate = gates[gcfg.id]
            stats = monitor.stats.get(gcfg.id)
            if not stats:
                continue

            # Temperature circuit breaker (per-GPU)
            if stats.temp_c >= gcfg.temp_pause_c:
                gate.trip_temp()
            elif stats.temp_c <= gcfg.temp_resume_c:
                gate.clear_temp()

            # VRAM warning
            if stats.vram_free_mb < 2048 and stats.vram_free_mb > 0:
                logging.warning(
                    f"[{gcfg.name}] Low VRAM: {stats.vram_free_mb:.0f}MB free "
                    f"of {stats.vram_total_mb:.0f}MB"
                )

        # Combined power circuit breaker (all GPUs)
        if combined >= config.combined_power_threshold_w:
            for gate in gates.values():
                gate.trip_power()
        else:
            for gate in gates.values():
                gate.clear_power()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_all(config: Config):
    """Start all proxy servers, monitor, and management API."""

    # Set up GPU monitor
    gpu_ids = [g.id for g in config.gpus]
    monitor = GpuMonitor(gpu_ids, config.monitor_interval_s)
    await monitor.start()

    # Set up request logger
    req_logger = RequestLogger(config.log_file)
    await req_logger.start()

    # Create gates and proxy apps
    gates: dict[int, GpuGate] = {}
    runners = []

    for gcfg in config.gpus:
        gate = GpuGate(gcfg)
        gates[gcfg.id] = gate

        app = make_gpu_app(gate, monitor, config, req_logger)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", gcfg.listen_port)
        await site.start()
        runners.append(runner)
        logging.info(f"[{gcfg.name}] Proxy listening on 0.0.0.0:{gcfg.listen_port} -> 127.0.0.1:{gcfg.backend_port}")

    # Management API
    mgmt_app = make_management_app(gates, monitor, config)
    mgmt_runner = web.AppRunner(mgmt_app)
    await mgmt_runner.setup()
    mgmt_site = web.TCPSite(mgmt_runner, "127.0.0.1", config.management_port)
    await mgmt_site.start()
    runners.append(mgmt_runner)
    logging.info(f"Management API on 0.0.0.0:{config.management_port}")

    # Start circuit breaker loop
    cb_task = asyncio.create_task(circuit_breaker_loop(gates, monitor, config))

    logging.info("=== AI Shaman is guarding your GPUs ===")

    # Wait for shutdown signal (cross-platform)
    stop_event = asyncio.Event()

    def _signal_handler(*args):
        logging.info("Shutdown signal received")
        stop_event.set()

    if sys.platform != "win32":
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)
    else:
        # Windows: use signal.signal() â€” works for SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    await stop_event.wait()

    # Cleanup
    logging.info("Shutting down AI Shaman...")
    cb_task.cancel()
    await monitor.stop()
    await req_logger.stop()
    for runner in runners:
        await runner.cleanup()
    logging.info("Shutdown complete.")


def main():
    # Find config
    config_path = os.environ.get(
        "GUARDIAN_CONFIG",
        str(Path(__file__).parent / "guardian_config.json"),
    )
    config = Config.from_file(config_path)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.log_file),
        ],
    )
    logging.info(f"Loading config from {config_path}")
    logging.info(f"GPUs: {[g.name for g in config.gpus]}")
    logging.info(f"Combined power threshold: {config.combined_power_threshold_w}W")

    asyncio.run(run_all(config))


if __name__ == "__main__":
    main()

