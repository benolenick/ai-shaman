#!/usr/bin/env python3
"""
AI Shaman -- Ollama GPU Guardian
a'shamon: "Guardian" in the Old Tongue

Async reverse proxy that sits between consumers and Ollama instances,
enforcing power budgets, temperature circuit breakers, and request
queuing to prevent simultaneous GPU power spikes that cause hard reboots.

Zero-change deployment: consumers hit the same ports as before.

Priority Job Queue (v2):
  Consumers submit jobs via POST /queue/submit with a priority level.
  The queue processes jobs in priority order: critical > normal > bulk.
  Results are stored in-memory and retrieved via GET /queue/result/{job_id}.
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
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

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
    vram_min_free_mb: float = 2000.0     # minimum free VRAM to accept requests (prevents OOM)
    ram_min_free_gb: float = 4.0         # minimum free system RAM to accept requests


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
    # Job queue config
    job_result_ttl_s: float = 600.0      # how long to keep results in memory
    bulk_inter_job_delay_s: float = 2.0  # delay between bulk jobs
    bulk_slow_backoff_s: float = 5.0     # base backoff when Ollama is slow (>60s)
    # Resource guard — prevents OOM crashes that require physical power cycle
    resource_guard_enabled: bool = True
    ram_min_free_gb: float = 4.0         # system-wide RAM floor

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path) as f:
            raw = json.load(f)
        gpu_dicts = raw.get("gpus", [])
        gpus = []
        for g in gpu_dicts:
            allowed = g.pop("allowed_models", [])
            vram_min = g.pop("vram_min_free_mb", 2000.0)
            ram_min = g.pop("ram_min_free_gb", 4.0)
            gc = GpuConfig(**g)
            gc.allowed_models = allowed
            gc.vram_min_free_mb = vram_min
            gc.ram_min_free_gb = ram_min
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
            job_result_ttl_s=raw.get("job_result_ttl_s", 600.0),
            bulk_inter_job_delay_s=raw.get("bulk_inter_job_delay_s", 2.0),
            bulk_slow_backoff_s=raw.get("bulk_slow_backoff_s", 5.0),
            resource_guard_enabled=raw.get("resource_guard_enabled", True),
            ram_min_free_gb=raw.get("ram_min_free_gb", 4.0),
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
            logging.error("nvidia-smi timed out (>10s) -- GPU may be hung")
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
        self.maintenance_locked = False  # hard block during model swap/maintenance
        self.maintenance_reason = ""
        self.temp_tripped = False     # auto-pause from temperature
        self.power_tripped = False    # auto-pause from combined power
        self.maintenance_locked = False  # hard block during model swap/maintenance
        self.maintenance_reason = ""
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
            logging.warning(f"[{self.cfg.name}] TEMP CIRCUIT BREAKER TRIPPED -- GPU {self.cfg.id} too hot")

    def clear_temp(self):
        if self.temp_tripped:
            self.temp_tripped = False
            if not self.paused and not self.power_tripped and not self.maintenance_locked:
                self._resume_event.set()
            logging.info(f"[{self.cfg.name}] Temp circuit breaker cleared")

    def trip_power(self):
        if not self.power_tripped:
            self.power_tripped = True
            self._resume_event.clear()
            logging.warning(f"[{self.cfg.name}] POWER BUDGET EXCEEDED -- queuing requests")

    def clear_power(self):
        if self.power_tripped:
            self.power_tripped = False
            if not self.paused and not self.temp_tripped and not self.maintenance_locked:
                self._resume_event.set()
            logging.info(f"[{self.cfg.name}] Power budget cleared")

    def manual_pause(self):
        self.paused = True
        self._resume_event.clear()
        logging.info(f"[{self.cfg.name}] Manually paused")

    def manual_resume(self):
        self.paused = False
        if not self.temp_tripped and not self.power_tripped and not self.maintenance_locked:
            self._resume_event.set()
        logging.info(f"[{self.cfg.name}] Manually resumed")

    def maintenance_lock(self, reason: str = ""):
        self.maintenance_locked = True
        self.maintenance_reason = reason or ""
        self._resume_event.clear()
        logging.info(
            f"[{self.cfg.name}] Maintenance lock engaged"
            + (f": {self.maintenance_reason}" if self.maintenance_reason else "")
        )

    def maintenance_unlock(self):
        self.maintenance_locked = False
        self.maintenance_reason = ""
        if not self.paused and not self.temp_tripped and not self.power_tripped:
            self._resume_event.set()
        logging.info(f"[{self.cfg.name}] Maintenance lock released")

    @property
    def is_blocked(self) -> bool:
        return self.paused or self.temp_tripped or self.power_tripped or self.maintenance_locked

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
# Priority Job Queue
# ---------------------------------------------------------------------------

# Priority ordinal (lower number = higher priority, for asyncio.PriorityQueue)
PRIORITY_ORDINAL = {"critical": 0, "normal": 1, "bulk": 2}
PRIORITY_NAMES = set(PRIORITY_ORDINAL.keys())

# Job states
JOB_PENDING   = "pending"
JOB_RUNNING   = "running"
JOB_DONE      = "done"
JOB_ERROR     = "error"
JOB_CANCELLED = "cancelled"


@dataclass
class QueueJob:
    job_id: str
    priority: str          # "critical" | "normal" | "bulk"
    consumer: str          # name of submitting consumer
    model: str
    prompt: str
    endpoint: str          # "/api/generate" | "/api/chat" | "/v1/chat/completions"
    extra: dict            # additional body fields (stream must be False)
    timeout: float
    submitted_at: float
    state: str = JOB_PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "priority": self.priority,
            "consumer": self.consumer,
            "model": self.model,
            "endpoint": self.endpoint,
            "state": self.state,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wait_s": round((self.started_at or time.time()) - self.submitted_at, 2),
            "latency_s": round((self.finished_at - self.started_at), 2) if self.finished_at and self.started_at else None,
            "error": self.error,
        }


class PriorityJobQueue:
    """
    In-memory priority job queue for Ollama inference requests.

    Design:
    - asyncio.PriorityQueue holds (ordinal, seq, job_id) tuples
    - Jobs dict holds QueueJob objects
    - Worker coroutine processes one job at a time in priority order
    - Higher-priority jobs inserted while bulk runs will be served first on next slot
    - Results stored with TTL; expired results cleaned up periodically
    """

    def __init__(self, config: Config):
        self.config = config
        self._pq: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._jobs: dict[str, QueueJob] = {}
        self._seq = 0
        self._worker_task: Optional[asyncio.Task] = None
        self._cleaner_task: Optional[asyncio.Task] = None
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # starts unpaused
        self._active_job_id: Optional[str] = None

        # Stats
        self._stats_by_consumer: dict[str, dict] = {}
        self._stats_by_priority: dict[str, dict] = {p: {"submitted": 0, "done": 0, "errors": 0, "total_latency_s": 0.0} for p in PRIORITY_NAMES}
        self._total_jobs = 0

        # backend URL: the queue worker calls Ollama directly on backend port
        # We pick the first GPU's backend for simplicity -- can be extended per-model
        self._backend_port: Optional[int] = None  # set in start()

    def start(self, backend_port: int):
        """Call from asyncio context after event loop is running."""
        self._backend_port = backend_port
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._cleaner_task = asyncio.create_task(self._cleanup_loop())
        logging.info(f"[JobQueue] Started, backend port {backend_port}")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
        if self._cleaner_task:
            self._cleaner_task.cancel()

    def pause(self):
        self._paused = True
        self._pause_event.clear()
        logging.info("[JobQueue] Paused")

    def resume(self):
        self._paused = False
        self._pause_event.set()
        logging.info("[JobQueue] Resumed")

    def submit(self, priority: str, consumer: str, model: str, prompt: str,
               endpoint: str, extra: dict, timeout: float) -> str:
        if priority not in PRIORITY_NAMES:
            raise ValueError(f"Invalid priority '{priority}'. Must be one of: {sorted(PRIORITY_NAMES)}")
        job_id = str(uuid.uuid4())
        job = QueueJob(
            job_id=job_id,
            priority=priority,
            consumer=consumer,
            model=model,
            prompt=prompt,
            endpoint=endpoint,
            extra=extra,
            timeout=timeout,
            submitted_at=time.time(),
        )
        self._jobs[job_id] = job
        ordinal = PRIORITY_ORDINAL[priority]
        self._seq += 1
        self._pq.put_nowait((ordinal, self._seq, job_id))
        self._total_jobs += 1

        # Stats
        self._stats_by_priority[priority]["submitted"] += 1
        if consumer not in self._stats_by_consumer:
            self._stats_by_consumer[consumer] = {"submitted": 0, "done": 0, "errors": 0, "total_latency_s": 0.0}
        self._stats_by_consumer[consumer]["submitted"] += 1

        logging.info(f"[JobQueue] Submitted job {job_id} priority={priority} consumer={consumer} model={model}")
        return job_id

    def get_result(self, job_id: str) -> Optional[QueueJob]:
        return self._jobs.get(job_id)

    def clear_priority(self, priority: str) -> int:
        """Mark all pending jobs of a given priority as cancelled. Returns count."""
        if priority not in PRIORITY_NAMES:
            raise ValueError(f"Invalid priority '{priority}'")
        count = 0
        for job in self._jobs.values():
            if job.priority == priority and job.state == JOB_PENDING:
                job.state = JOB_CANCELLED
                job.finished_at = time.time()
                job.error = "Cancelled by operator"
                count += 1
        # Drain and re-insert non-cancelled jobs
        # (PriorityQueue doesn't support item removal, so we drain+refill)
        remaining = []
        while not self._pq.empty():
            try:
                item = self._pq.get_nowait()
                jid = item[2]
                j = self._jobs.get(jid)
                if j and j.state == JOB_PENDING:
                    remaining.append(item)
            except asyncio.QueueEmpty:
                break
        for item in remaining:
            self._pq.put_nowait(item)
        logging.info(f"[JobQueue] Cleared {count} pending {priority} jobs")
        return count

    def status(self) -> dict:
        pending_by_priority = {p: 0 for p in PRIORITY_NAMES}
        pending_list = []
        for job in self._jobs.values():
            if job.state == JOB_PENDING:
                pending_by_priority[job.priority] += 1
                pending_list.append(job.to_summary())
        pending_list.sort(key=lambda j: (PRIORITY_ORDINAL.get(j["priority"], 99), j["submitted_at"]))

        active = None
        if self._active_job_id:
            j = self._jobs.get(self._active_job_id)
            if j:
                active = j.to_summary()

        return {
            "paused": self._paused,
            "active_job": active,
            "queue_depth": sum(pending_by_priority.values()),
            "pending_by_priority": pending_by_priority,
            "pending_jobs": pending_list,
            "total_jobs_submitted": self._total_jobs,
        }

    def stats(self) -> dict:
        def _avg(d: dict) -> float:
            done = d.get("done", 0)
            return round(d.get("total_latency_s", 0.0) / done, 2) if done else 0.0

        by_priority = {}
        for p, s in self._stats_by_priority.items():
            by_priority[p] = {**s, "avg_latency_s": _avg(s)}

        by_consumer = {}
        for c, s in self._stats_by_consumer.items():
            by_consumer[c] = {**s, "avg_latency_s": _avg(s)}

        return {
            "total_jobs": self._total_jobs,
            "by_priority": by_priority,
            "by_consumer": by_consumer,
        }

    async def _worker_loop(self):
        """Process jobs from the priority queue one at a time."""
        logging.info("[JobQueue] Worker started")
        last_priority = None
        bulk_backoff = 0.0

        while True:
            # Wait if paused
            await self._pause_event.wait()

            try:
                # Get next job (timeout 1s so we can check pause state)
                try:
                    item = await asyncio.wait_for(self._pq.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                _ordinal, _seq, job_id = item
                job = self._jobs.get(job_id)
                if not job or job.state != JOB_PENDING:
                    # Job was cancelled or already gone
                    continue

                # Check if a higher-priority job is in queue while we're about to run bulk
                if job.priority == "bulk":
                    # Peek: if critical/normal is waiting, re-queue this bulk and loop
                    if self._has_higher_priority("bulk"):
                        self._pq.put_nowait((_ordinal, _seq, job_id))
                        await asyncio.sleep(0.1)
                        continue

                # Inter-job spacing for bulk (to avoid hammering Ollama)
                if job.priority == "bulk" and last_priority == "bulk":
                    delay = max(self.config.bulk_inter_job_delay_s, bulk_backoff)
                    if delay > 0:
                        logging.debug(f"[JobQueue] Bulk inter-job delay {delay:.1f}s")
                        await asyncio.sleep(delay)

                # Run the job
                self._active_job_id = job_id
                job.state = JOB_RUNNING
                job.started_at = time.time()
                logging.info(f"[JobQueue] Running job {job_id} priority={job.priority} consumer={job.consumer}")

                try:
                    result = await self._execute_job(job)
                    job.result = result
                    job.state = JOB_DONE
                    job.finished_at = time.time()
                    latency = job.finished_at - job.started_at

                    # Bulk backoff: if response took >60s, back off exponentially
                    if job.priority == "bulk" and latency > 60.0:
                        bulk_backoff = min(bulk_backoff * 2 + self.config.bulk_slow_backoff_s, 60.0)
                        logging.info(f"[JobQueue] Slow Ollama ({latency:.0f}s), bulk backoff -> {bulk_backoff:.0f}s")
                    else:
                        bulk_backoff = max(0.0, bulk_backoff - 1.0)  # decay

                    self._stats_by_priority[job.priority]["done"] += 1
                    self._stats_by_priority[job.priority]["total_latency_s"] += latency
                    self._stats_by_consumer[job.consumer]["done"] += 1
                    self._stats_by_consumer[job.consumer]["total_latency_s"] += latency
                    logging.info(f"[JobQueue] Job {job_id} done in {latency:.1f}s")

                except Exception as e:
                    job.state = JOB_ERROR
                    job.finished_at = time.time()
                    job.error = str(e)
                    self._stats_by_priority[job.priority]["errors"] += 1
                    self._stats_by_consumer[job.consumer]["errors"] += 1
                    logging.warning(f"[JobQueue] Job {job_id} failed: {e}")

                last_priority = job.priority
                self._active_job_id = None

            except asyncio.CancelledError:
                logging.info("[JobQueue] Worker cancelled")
                break
            except Exception as e:
                logging.error(f"[JobQueue] Worker error: {e}")
                await asyncio.sleep(1.0)

    def _has_higher_priority(self, than_priority: str) -> bool:
        """Check if there are higher-priority jobs pending in the queue."""
        ordinal = PRIORITY_ORDINAL.get(than_priority, 99)
        for job in self._jobs.values():
            if job.state == JOB_PENDING and PRIORITY_ORDINAL.get(job.priority, 99) < ordinal:
                return True
        return False

    async def _execute_job(self, job: QueueJob) -> dict:
        """Execute a single job against Ollama backend."""
        if not self._backend_port:
            raise RuntimeError("Job queue has no backend port configured")

        url = f"http://127.0.0.1:{self._backend_port}{job.endpoint}"

        # Build request body
        body: dict[str, Any] = {**job.extra}
        body["model"] = job.model
        body["stream"] = False  # always non-streaming for queue jobs

        # Set prompt based on endpoint
        if job.endpoint in {"/api/generate"}:
            body["prompt"] = job.prompt
        elif job.endpoint in {"/api/chat"}:
            # If extra has messages, use them; otherwise wrap prompt
            if "messages" not in body:
                body["messages"] = [{"role": "user", "content": job.prompt}]
        elif job.endpoint == "/v1/chat/completions":
            if "messages" not in body:
                body["messages"] = [{"role": "user", "content": job.prompt}]

        timeout = aiohttp.ClientTimeout(total=job.timeout, sock_read=job.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body) as resp:
                resp_body = await resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned {resp.status}: {resp_body[:200].decode('utf-8', errors='replace')}")
                try:
                    return json.loads(resp_body.decode("utf-8"))
                except Exception:
                    return {"raw": resp_body.decode("utf-8", errors="replace")}

    async def _cleanup_loop(self):
        """Periodically remove expired results to prevent memory leak."""
        while True:
            try:
                await asyncio.sleep(60.0)
                now = time.time()
                ttl = self.config.job_result_ttl_s
                to_delete = [
                    jid for jid, job in self._jobs.items()
                    if job.state in {JOB_DONE, JOB_ERROR, JOB_CANCELLED}
                    and job.finished_at is not None
                    and (now - job.finished_at) > ttl
                ]
                for jid in to_delete:
                    del self._jobs[jid]
                if to_delete:
                    logging.debug(f"[JobQueue] Cleaned {len(to_delete)} expired job results")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.warning(f"[JobQueue] Cleanup error: {e}")


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


# Dangerous admin endpoints -- blocked entirely at the proxy
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
    for p in parts[1:]:
        if p == out[-1]:
            continue
        out.append(p)
    cleaned = " ".join(out).strip()
    # Trim dangling unfinished tails from token cutoffs.
    cleaned = re.sub(r"(The answer is|Answer:)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _reasoning_fallback(reasoning: str) -> str:
    r = (reasoning or "").strip()
    if not r:
        return ""
    # Try to recover an explicit "answer is ..." style sentence first.
    m = re.findall(r"([^.?!]*answer[^.?!]*[.?!])", r, flags=re.IGNORECASE)
    if m:
        return _dedupe_repeated_sentences(m[-1]).strip()
    # Otherwise use the last substantial sentence.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", r) if p.strip()]
    if parts:
        return _dedupe_repeated_sentences(parts[-1]).strip()
    return ""


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
            reasoning = str(msg.get("reasoning") or "")
            msg.pop("reasoning", None)
            content = msg.get("content")
            if isinstance(content, str):
                cleaned = _dedupe_repeated_sentences(content)
                if not cleaned and reasoning:
                    cleaned = _reasoning_fallback(reasoning)
                msg["content"] = cleaned

    # Ollama /api/chat non-stream JSON shape
    if path == "/api/chat":
        msg = payload.get("message")
        if isinstance(msg, dict):
            thinking = str(msg.get("thinking") or "")
            msg.pop("thinking", None)
            content = msg.get("content")
            if isinstance(content, str):
                cleaned = _dedupe_repeated_sentences(content)
                if not cleaned and thinking:
                    cleaned = _reasoning_fallback(thinking)
                msg["content"] = cleaned

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
        model = str(data.get("model") or "").lower()
        if model.startswith("qwen"):
            mt = data.get("max_tokens")
            if not isinstance(mt, int) or mt < 512:
                data["max_tokens"] = 512

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
            "allowed_models": gcfg.allowed_models or ["(any -- no restriction)"],
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
        if gate.maintenance_locked:
            gate.total_requests += 1
            gate.total_errors += 1
            req_logger.log(
                gate.cfg.name,
                method,
                path,
                503,
                0.0,
                gpu_stats.power_w if gpu_stats else 0.0,
                "",
            )
            return web.json_response({
                "error": "GPU PORT LOCKED FOR MAINTENANCE",
                "guardian": "ai-shaman",
                "gpu": {
                    "id": gate.cfg.id,
                    "name": gate.cfg.name,
                    "port": gate.cfg.listen_port,
                },
                "message": (
                    "This GPU proxy is temporarily locked to prevent requests "
                    "during model swap or maintenance."
                ),
                "reason": gate.maintenance_reason or "model swap in progress",
            }, status=503)

        # Maintenance lock: hard block all traffic
        if gate.maintenance_locked:
            gate.total_requests += 1
            gate.total_errors += 1
            req_logger.log(
                gate.cfg.name,
                method,
                path,
                503,
                0.0,
                gpu_stats.power_w if gpu_stats else 0.0,
                "",
            )
            return web.json_response({
                "error": "GPU PORT LOCKED FOR MAINTENANCE",
                "guardian": "ai-shaman",
                "gpu": {
                    "id": gate.cfg.id,
                    "name": gate.cfg.name,
                    "port": gate.cfg.listen_port,
                },
                "message": (
                    "This GPU proxy is temporarily locked to prevent requests "
                    "during model swap or maintenance."
                ),
                "reason": gate.maintenance_reason or "model swap in progress",
            }, status=503)

        # -- BLOCK DANGEROUS ADMIN ENDPOINTS -----------------------------------
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
        # ----------------------------------------------------------------------

        # Extract model for logging and affinity check
        model = ""
        if method == "POST" and _is_queued(path):
            model = await _extract_model(request)

        # -- MODEL AFFINITY CHECK ---------------------------------------------
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
                    f"[{gate.cfg.name}] MODEL REJECTED: empty/missing model field -- returning 403"
                )
                req_logger.log(gate.cfg.name, method, path, 403, 0.0,
                               gpu_stats.power_w if gpu_stats else 0.0, "")
                return web.json_response(rejection, status=403)

            if not _check_model_allowed(model, gate.cfg):
                gate.total_requests += 1
                gate.total_errors += 1
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
                    f"{gate.cfg.allowed_models} -- returning 403"
                )
                req_logger.log(gate.cfg.name, method, path, 403, 0.0,
                               gpu_stats.power_w if gpu_stats else 0.0, model)
                return web.json_response(rejection, status=403)
        # ----------------------------------------------------------------------

        # -- RESOURCE GUARD (prevents OOM crashes) -----------------------------
        if method == "POST" and _is_queued(path) and config.resource_guard_enabled:
            # Check VRAM
            if gpu_stats and gpu_stats.vram_free_mb > 0:
                if gpu_stats.vram_free_mb < gate.cfg.vram_min_free_mb:
                    gate.total_requests += 1
                    gate.total_errors += 1
                    logging.warning(
                        f"[{gate.cfg.name}] RESOURCE GUARD: VRAM too low "
                        f"({gpu_stats.vram_free_mb:.0f}MB free < "
                        f"{gate.cfg.vram_min_free_mb:.0f}MB minimum) -- "
                        f"rejecting request to prevent OOM crash"
                    )
                    req_logger.log(gate.cfg.name, method, path, 503, 0.0,
                                   gpu_stats.power_w, model)
                    return web.json_response({
                        "error": "RESOURCE GUARD: INSUFFICIENT VRAM",
                        "guardian": "ai-shaman",
                        "gpu": {
                            "id": gate.cfg.id,
                            "name": gate.cfg.name,
                            "vram_free_mb": gpu_stats.vram_free_mb,
                            "vram_total_mb": gpu_stats.vram_total_mb,
                            "vram_min_free_mb": gate.cfg.vram_min_free_mb,
                        },
                        "message": (
                            f"Request rejected to prevent OOM crash. "
                            f"GPU {gate.cfg.id} has only {gpu_stats.vram_free_mb:.0f}MB "
                            f"free VRAM (minimum: {gate.cfg.vram_min_free_mb:.0f}MB). "
                            f"Wait for current workload to finish or reduce model size."
                        ),
                    }, status=503)

            # Check system RAM
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            avail_gb = int(line.split()[1]) / (1024 * 1024)
                            if avail_gb < config.ram_min_free_gb:
                                gate.total_requests += 1
                                gate.total_errors += 1
                                logging.warning(
                                    f"[{gate.cfg.name}] RESOURCE GUARD: System RAM too low "
                                    f"({avail_gb:.1f}GB free < {config.ram_min_free_gb:.1f}GB minimum) -- "
                                    f"rejecting to prevent system freeze"
                                )
                                req_logger.log(gate.cfg.name, method, path, 503, 0.0,
                                               gpu_stats.power_w if gpu_stats else 0.0, model)
                                return web.json_response({
                                    "error": "RESOURCE GUARD: INSUFFICIENT SYSTEM RAM",
                                    "guardian": "ai-shaman",
                                    "ram_free_gb": round(avail_gb, 1),
                                    "ram_min_free_gb": config.ram_min_free_gb,
                                    "message": (
                                        f"Request rejected to prevent system freeze. "
                                        f"Only {avail_gb:.1f}GB RAM available "
                                        f"(minimum: {config.ram_min_free_gb:.1f}GB). "
                                        f"Kill other processes or wait for memory to free."
                                    ),
                                }, status=503)
                            break
            except Exception:
                pass  # /proc/meminfo not available (non-Linux) — skip check
        # ----------------------------------------------------------------------

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
                        config: Config, job_queue: PriorityJobQueue) -> web.Application:
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
                "maintenance_locked": gate.maintenance_locked,
                "maintenance_reason": gate.maintenance_reason,
            })
        return web.json_response({
            "guardian": "AI Shaman",
            "version": "2.0.0",
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

    async def maintenance_lock_handler(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            data = {}
        gpu_id = data.get("gpu_id")
        reason = data.get("reason", "model swap in progress")
        if gpu_id is None:
            for gate in gates.values():
                gate.maintenance_lock(reason)
            return web.json_response({
                "status": "maintenance lock enabled for all GPUs",
                "reason": reason,
            })
        if gpu_id not in gates:
            return web.json_response({"error": f"Unknown GPU {gpu_id}"}, status=404)
        gates[gpu_id].maintenance_lock(reason)
        return web.json_response({
            "status": f"maintenance lock enabled for GPU {gpu_id}",
            "reason": reason,
        })

    async def maintenance_unlock_handler(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            data = {}
        gpu_id = data.get("gpu_id")
        if gpu_id is None:
            for gate in gates.values():
                gate.maintenance_unlock()
            return web.json_response({"status": "maintenance lock disabled for all GPUs"})
        if gpu_id not in gates:
            return web.json_response({"error": f"Unknown GPU {gpu_id}"}, status=404)
        gates[gpu_id].maintenance_unlock()
        return web.json_response({"status": f"maintenance lock disabled for GPU {gpu_id}"})

    # -------------------------------------------------------------------------
    # Job Queue endpoints
    # -------------------------------------------------------------------------

    async def queue_submit_handler(request: web.Request) -> web.Response:
        """POST /queue/submit -- submit a job to the priority queue."""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON body"}, status=400)

        priority = data.get("priority", "normal")
        consumer = data.get("consumer", "unknown")
        model = data.get("model", "")
        prompt = data.get("prompt", "")
        endpoint = data.get("endpoint", "/api/generate")
        timeout = float(data.get("timeout", config.queue_timeout_s))
        # Extra fields forwarded to Ollama (e.g. options, system, messages)
        extra = {k: v for k, v in data.items()
                 if k not in {"priority", "consumer", "model", "prompt", "endpoint", "timeout"}}

        if not model:
            return web.json_response({"error": "Missing required field: model"}, status=400)
        if not prompt and "messages" not in extra:
            return web.json_response({"error": "Missing required field: prompt (or messages in extra)"}, status=400)
        if priority not in PRIORITY_NAMES:
            return web.json_response(
                {"error": f"Invalid priority '{priority}'. Must be one of: {sorted(PRIORITY_NAMES)}"},
                status=400,
            )
        if endpoint not in QUEUED_PATHS:
            return web.json_response(
                {"error": f"Invalid endpoint '{endpoint}'. Must be one of: {sorted(QUEUED_PATHS)}"},
                status=400,
            )

        try:
            job_id = job_queue.submit(
                priority=priority,
                consumer=consumer,
                model=model,
                prompt=prompt,
                endpoint=endpoint,
                extra=extra,
                timeout=timeout,
            )
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        return web.json_response({
            "ok": True,
            "job_id": job_id,
            "priority": priority,
            "consumer": consumer,
            "model": model,
            "endpoint": endpoint,
            "poll_url": f"/queue/result/{job_id}",
        }, status=202)

    async def queue_result_handler(request: web.Request) -> web.Response:
        """GET /queue/result/{job_id} -- poll for job result."""
        job_id = request.match_info.get("job_id", "")
        job = job_queue.get_result(job_id)
        if not job:
            return web.json_response(
                {"error": f"Job not found: {job_id}", "hint": "Job may have expired or never existed."},
                status=404,
            )
        resp_data = job.to_summary()
        # Include result/error only when done
        if job.state == JOB_DONE:
            resp_data["result"] = job.result
            return web.json_response(resp_data, status=200)
        elif job.state == JOB_ERROR:
            return web.json_response(resp_data, status=200)
        elif job.state in {JOB_PENDING, JOB_RUNNING}:
            # Not done yet -- return 202 with Retry-After
            headers = {"Retry-After": "2"}
            return web.json_response(resp_data, status=202, headers=headers)
        else:
            return web.json_response(resp_data, status=200)

    async def queue_status_handler(request: web.Request) -> web.Response:
        """GET /queue/status -- show queue state."""
        return web.json_response(job_queue.status())

    async def queue_pause_handler(request: web.Request) -> web.Response:
        """POST /queue/pause -- pause queue processing."""
        job_queue.pause()
        return web.json_response({"status": "queue paused"})

    async def queue_resume_handler(request: web.Request) -> web.Response:
        """POST /queue/resume -- resume queue processing."""
        job_queue.resume()
        return web.json_response({"status": "queue resumed"})

    async def queue_clear_handler(request: web.Request) -> web.Response:
        """POST /queue/clear -- clear pending jobs by priority."""
        try:
            data = await request.json()
        except Exception:
            data = {}
        priority = data.get("priority")
        if not priority:
            return web.json_response(
                {"error": "Missing required field: priority (critical|normal|bulk)"},
                status=400,
            )
        try:
            count = job_queue.clear_priority(priority)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"status": f"cleared {count} pending {priority} jobs", "count": count})

    async def queue_stats_handler(request: web.Request) -> web.Response:
        """GET /queue/stats -- throughput and latency stats."""
        return web.json_response(job_queue.stats())

    app = web.Application()
    app.router.add_get("/guardian/status", status_handler)
    app.router.add_get("/guardian/metrics", metrics_handler)
    app.router.add_post("/guardian/pause", pause_handler)
    app.router.add_post("/guardian/resume", resume_handler)
    app.router.add_post("/guardian/maintenance/lock", maintenance_lock_handler)
    app.router.add_post("/guardian/maintenance/unlock", maintenance_unlock_handler)
    # Job queue endpoints
    app.router.add_post("/queue/submit", queue_submit_handler)
    app.router.add_get("/queue/result/{job_id}", queue_result_handler)
    app.router.add_get("/queue/status", queue_status_handler)
    app.router.add_post("/queue/pause", queue_pause_handler)
    app.router.add_post("/queue/resume", queue_resume_handler)
    app.router.add_post("/queue/clear", queue_clear_handler)
    app.router.add_get("/queue/stats", queue_stats_handler)
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

    # Job queue (uses first GPU's backend by default; WTS targets model via submit body)
    job_queue = PriorityJobQueue(config)
    primary_backend_port = config.gpus[0].backend_port if config.gpus else 11444
    job_queue.start(primary_backend_port)

    # Management API (now includes job queue)
    mgmt_app = make_management_app(gates, monitor, config, job_queue)
    mgmt_runner = web.AppRunner(mgmt_app)
    await mgmt_runner.setup()
    mgmt_site = web.TCPSite(mgmt_runner, "0.0.0.0", config.management_port)
    await mgmt_site.start()
    runners.append(mgmt_runner)
    logging.info(f"Management API on 0.0.0.0:{config.management_port}")

    # Start circuit breaker loop
    cb_task = asyncio.create_task(circuit_breaker_loop(gates, monitor, config))

    logging.info("=== AI Shaman v2.0 is guarding your GPUs ===")

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
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

    await stop_event.wait()

    # Cleanup
    logging.info("Shutting down AI Shaman...")
    cb_task.cancel()
    await job_queue.stop()
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
