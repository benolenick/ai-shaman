#!/usr/bin/env python3
"""
AI Shaman operator CLI.

Human mode:
  aishaman

Automation mode:
  aishaman <subcommand> [flags] --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_MGMT_URL = "http://127.0.0.1:11450"


class CliError(Exception):
    pass


@dataclass
class ApiResult:
    status: int
    data: Any


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> ApiResult:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace").strip()
            data = json.loads(text) if text else {}
            return ApiResult(status=resp.getcode(), data=data)
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace").strip()
        try:
            data = json.loads(raw) if raw else {"error": f"HTTP {e.code}"}
        except json.JSONDecodeError:
            data = {"error": raw or f"HTTP {e.code}"}
        return ApiResult(status=e.code, data=data)
    except urllib.error.URLError as e:
        raise CliError(f"Network error calling {url}: {e}") from e


class AiShamanClient:
    def __init__(self, management_url: str):
        self.management_url = management_url.rstrip("/")

    def status(self) -> dict[str, Any]:
        res = _http_json("GET", f"{self.management_url}/guardian/status")
        if res.status != 200:
            raise CliError(f"Status request failed: {res.status} {res.data}")
        return res.data

    def health(self) -> dict[str, Any]:
        st = self.status()
        gpus = st.get("gpus", [])
        issues = []
        for gpu in gpus:
            if gpu.get("temp_tripped"):
                issues.append(f"GPU {gpu.get('id')} temp breaker tripped")
            if gpu.get("power_tripped"):
                issues.append(f"GPU {gpu.get('id')} power breaker tripped")
            if gpu.get("maintenance_locked"):
                issues.append(f"GPU {gpu.get('id')} maintenance locked")
        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "combined_power_w": st.get("combined_power_w"),
            "combined_power_threshold_w": st.get("combined_power_threshold_w"),
            "gpus": gpus,
        }

    def lock(self, gpu_id: int | None, reason: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason": reason}
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        res = _http_json("POST", f"{self.management_url}/guardian/maintenance/lock", payload)
        if res.status != 200:
            raise CliError(f"Lock failed: {res.status} {res.data}")
        return res.data

    def unlock(self, gpu_id: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        res = _http_json("POST", f"{self.management_url}/guardian/maintenance/unlock", payload)
        if res.status != 200:
            raise CliError(f"Unlock failed: {res.status} {res.data}")
        return res.data


def _find_gpu(status: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    for gpu in status.get("gpus", []):
        if int(gpu.get("id", -1)) == gpu_id:
            return gpu
    raise CliError(f"Unknown GPU id: {gpu_id}")


def _backend_url_for_gpu(status: dict[str, Any], gpu_id: int, path: str) -> str:
    gpu = _find_gpu(status, gpu_id)
    port = gpu.get("backend_port")
    if not isinstance(port, int):
        raise CliError(f"GPU {gpu_id} has no backend_port in status payload")
    return f"http://127.0.0.1:{port}{path}"


def _models_list(status: dict[str, Any], gpu_id: int) -> dict[str, Any]:
    url = _backend_url_for_gpu(status, gpu_id, "/api/tags")
    res = _http_json("GET", url)
    if res.status != 200:
        raise CliError(f"List models failed: {res.status} {res.data}")
    return res.data


def _models_pull(status: dict[str, Any], gpu_id: int, model: str) -> dict[str, Any]:
    url = _backend_url_for_gpu(status, gpu_id, "/api/pull")
    res = _http_json("POST", url, {"name": model, "stream": False}, timeout=1200.0)
    if res.status != 200:
        raise CliError(f"Pull failed: {res.status} {res.data}")
    return res.data


def _models_delete(status: dict[str, Any], gpu_id: int, model: str) -> dict[str, Any]:
    url = _backend_url_for_gpu(status, gpu_id, "/api/delete")
    res = _http_json("DELETE", url, {"name": model})
    if res.status != 200:
        raise CliError(f"Delete failed: {res.status} {res.data}")
    return res.data


def _probe_model(status: dict[str, Any], gpu_id: int, model: str) -> dict[str, Any]:
    proxy_gpu = _find_gpu(status, gpu_id)
    listen_port = proxy_gpu.get("listen_port")
    if not isinstance(listen_port, int):
        raise CliError(f"GPU {gpu_id} has no listen_port in status payload")
    url = f"http://127.0.0.1:{listen_port}/api/generate"
    payload = {"model": model, "prompt": "health check", "stream": False}
    res = _http_json("POST", url, payload, timeout=180.0)
    if res.status != 200:
        raise CliError(f"Probe failed: {res.status} {res.data}")
    return res.data


def _print_result(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2))
        return
    for k, v in result.items():
        if isinstance(v, (dict, list)):
            print(f"{k}: {json.dumps(v, indent=2)}")
        else:
            print(f"{k}: {v}")


def _gpu_rows(status: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = status.get("gpus", [])
    rows: list[dict[str, Any]] = []
    for idx, gpu in enumerate(gpus, start=1):
        rows.append({
            "opt": idx,
            "id": gpu.get("id"),
            "name": gpu.get("name"),
            "listen_port": gpu.get("listen_port"),
            "backend_port": gpu.get("backend_port"),
            "allowed_models": gpu.get("allowed_models") or ["(any)"],
        })
    return rows


def _print_gpu_choices(status: dict[str, Any]) -> None:
    rows = _gpu_rows(status)
    if not rows:
        print("No GPUs reported by guardian.")
        return
    print("GPU choices:")
    for r in rows:
        print(
            f"  [{r['opt']}] id={r['id']}  {r['name']}  "
            f"proxy:{r['listen_port']} backend:{r['backend_port']}  "
            f"models:{','.join(r['allowed_models'])}"
        )


def _interactive_pick_gpu_id(status: dict[str, Any], allow_all: bool = False) -> int | None:
    rows = _gpu_rows(status)
    if not rows:
        raise CliError("No GPUs available from status")
    _print_gpu_choices(status)
    prompt = "Pick GPU option number or GPU id"
    if allow_all:
        prompt += " (or 'all')"
    prompt += ": "
    raw = input(prompt).strip().lower()
    if allow_all and raw in {"all", "*", ""}:
        return None
    if not raw:
        raise CliError("Selection required")

    for r in rows:
        if raw == str(r["opt"]) or raw == str(r["id"]):
            return int(r["id"])
    raise CliError(f"Unknown GPU selection: {raw}")


def _models_list_all(status: dict[str, Any]) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    for gpu in status.get("gpus", []):
        gid = int(gpu.get("id"))
        name = str(gpu.get("name"))
        tags = _models_list(status, gid)
        out.append({"gpu_id": gid, "gpu_name": name, "models": tags.get("models", [])})
    return {"gpus": out}


def _model_names(tags_payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for m in tags_payload.get("models", []):
        n = m.get("name")
        if isinstance(n, str) and n:
            names.add(n)
        n2 = m.get("model")
        if isinstance(n2, str) and n2:
            names.add(n2)
    return names


def _model_exists_on_gpu(status: dict[str, Any], gpu_id: int, model: str) -> bool:
    tags = _models_list(status, gpu_id)
    names = _model_names(tags)
    return model in names


def _swap_preflight(status: dict[str, Any], gpu_id: int, power_headroom_min: float = 0.10) -> dict[str, Any]:
    gpu = _find_gpu(status, gpu_id)
    issues: list[str] = []
    queue_depth = int(gpu.get("queue_depth", 0) or 0)
    if queue_depth > 0:
        issues.append(f"GPU {gpu_id} has queue_depth={queue_depth}")
    if bool(gpu.get("temp_tripped")):
        issues.append(f"GPU {gpu_id} temp circuit breaker is tripped")
    if bool(gpu.get("power_tripped")):
        issues.append(f"GPU {gpu_id} power circuit breaker is tripped")
    if bool(gpu.get("paused")):
        issues.append(f"GPU {gpu_id} is manually paused")
    if bool(gpu.get("maintenance_locked")):
        issues.append(f"GPU {gpu_id} is already maintenance locked")

    combined = float(status.get("combined_power_w", 0.0) or 0.0)
    threshold = float(status.get("combined_power_threshold_w", 0.0) or 0.0)
    if threshold > 0:
        ratio = combined / threshold
        if ratio >= (1.0 - power_headroom_min):
            issues.append(
                f"Combined power is high ({combined:.1f}W/{threshold:.1f}W, {ratio:.0%})"
            )

    return {
        "ok": len(issues) == 0,
        "gpu_id": gpu_id,
        "issues": issues,
        "combined_power_w": combined,
        "combined_power_threshold_w": threshold,
    }


def _run_swap(
    client: AiShamanClient,
    gpu_id: int,
    target_model: str,
    old_model: str | None,
    keep_old: bool,
    pull_mode: str,
    skip_probe: bool,
    reason: str,
    force: bool = False,
    confirm_download: Callable[[str, int], bool] | None = None,
) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    locked = False
    try:
        status = client.status()
        preflight = _swap_preflight(status, gpu_id)
        timeline.append({"step": "preflight", "status": "ok" if preflight["ok"] else "blocked", "result": preflight})
        if not preflight["ok"] and not force:
            return {
                "ok": False,
                "gpu_id": gpu_id,
                "target_model": target_model,
                "error": "Swap blocked by preflight safety checks. Re-run with --force to override.",
                "timeline": timeline,
            }

        timeline.append({"step": "lock", "status": "start"})
        lock_res = client.lock(gpu_id, reason)
        timeline.append({"step": "lock", "status": "ok", "result": lock_res})
        locked = True
        time.sleep(0.3)

        status = client.status()

        if pull_mode not in {"auto", "always", "never"}:
            raise CliError(f"Invalid pull_mode: {pull_mode}")

        should_pull = False
        if pull_mode == "always":
            should_pull = True
            timeline.append({"step": "pull_decision", "mode": "always", "decision": "pull"})
        elif pull_mode == "never":
            timeline.append({"step": "pull_decision", "mode": "never", "decision": "skip"})
        else:
            exists = _model_exists_on_gpu(status, gpu_id, target_model)
            if exists:
                timeline.append({
                    "step": "pull_decision",
                    "mode": "auto",
                    "decision": "skip",
                    "reason": "model already present on backend",
                })
            else:
                if confirm_download is not None:
                    approved = bool(confirm_download(target_model, gpu_id))
                    if not approved:
                        raise CliError("Swap cancelled by operator before model download")
                should_pull = True
                timeline.append({
                    "step": "pull_decision",
                    "mode": "auto",
                    "decision": "pull",
                    "reason": "model missing on backend",
                })

        if should_pull:
            timeline.append({"step": "pull", "status": "start", "model": target_model})
            pull_res = _models_pull(status, gpu_id, target_model)
            timeline.append({"step": "pull", "status": "ok", "result": pull_res})

        if not keep_old and old_model:
            timeline.append({"step": "delete_old", "status": "start", "model": old_model})
            del_res = _models_delete(status, gpu_id, old_model)
            timeline.append({"step": "delete_old", "status": "ok", "result": del_res})

        if not skip_probe:
            timeline.append({"step": "probe", "status": "start", "model": target_model})
            probe_res = _probe_model(status, gpu_id, target_model)
            timeline.append({"step": "probe", "status": "ok", "result": probe_res})

        unlock_res = client.unlock(gpu_id)
        locked = False
        timeline.append({"step": "unlock", "status": "ok", "result": unlock_res})

        return {"ok": True, "gpu_id": gpu_id, "target_model": target_model, "timeline": timeline}
    except Exception as e:
        fail = {"ok": False, "gpu_id": gpu_id, "target_model": target_model, "error": str(e), "timeline": timeline}
        if locked:
            try:
                unlock_res = client.unlock(gpu_id)
                fail["auto_unlock"] = {"ok": True, "result": unlock_res}
            except Exception as unlock_err:
                fail["auto_unlock"] = {"ok": False, "error": str(unlock_err)}
        return fail


def _interactive_menu(client: AiShamanClient) -> int:
    while True:
        print("\nAI Shaman Console")
        print("1. View status (temps, power, queue, lock states)")
        print("2. Health check (quick pass/fail + issue list)")
        print("3. Lock GPU (block requests on selected proxy port during maintenance)")
        print("4. Unlock GPU (resume requests after maintenance)")
        print("5. List models on GPU backend (what Ollama actually has on each card)")
        print("6. Guided swap workflow (lock -> pull -> optional delete old -> probe -> unlock)")
        print("7. Exit")
        try:
            choice = input("Choose [1-7] (or q to quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AI Shaman Console.")
            return 0
        try:
            if choice in {"7", "q", "quit", "exit"}:
                return 0
            if choice == "1":
                _print_result(client.status(), as_json=False)
            elif choice == "2":
                _print_result(client.health(), as_json=False)
            elif choice == "3":
                st = client.status()
                gpu_id = _interactive_pick_gpu_id(st, allow_all=True)
                reason = input("Reason [manual maintenance]: ").strip() or "manual maintenance"
                _print_result(client.lock(gpu_id, reason), as_json=False)
            elif choice == "4":
                st = client.status()
                gpu_id = _interactive_pick_gpu_id(st, allow_all=True)
                _print_result(client.unlock(gpu_id), as_json=False)
            elif choice == "5":
                st = client.status()
                print("Select one GPU, or type 'all' to list models for every backend.")
                sel = _interactive_pick_gpu_id(st, allow_all=True)
                if sel is None:
                    _print_result(_models_list_all(st), as_json=False)
                else:
                    _print_result(_models_list(st, sel), as_json=False)
            elif choice == "6":
                st = client.status()
                gpu_id = _interactive_pick_gpu_id(st, allow_all=False)
                target = input("Target model (e.g. qwen3.5:latest): ").strip()
                old = input("Old model to delete (optional): ").strip() or None
                keep_old = input("Keep old model? [y/N]: ").strip().lower() in {"y", "yes"}
                pm = (input("Pull mode [auto/always/never] (default auto): ").strip().lower() or "auto")
                if pm not in {"auto", "always", "never"}:
                    raise CliError("Pull mode must be one of: auto, always, never")
                skip_probe = input("Skip probe step? [y/N]: ").strip().lower() in {"y", "yes"}
                force = input("Force swap even if preflight warns? [y/N]: ").strip().lower() in {"y", "yes"}
                reason = input("Lock reason [guided swap]: ").strip() or "guided swap"

                def ask_download(model: str, gid: int) -> bool:
                    ans = input(
                        f"Model '{model}' is missing on GPU {gid}. Download now? [y/N]: "
                    ).strip().lower()
                    return ans in {"y", "yes"}

                res = _run_swap(
                    client,
                    gpu_id,
                    target,
                    old,
                    keep_old,
                    pm,
                    skip_probe,
                    reason,
                    force=force,
                    confirm_download=ask_download,
                )
                _print_result(res, as_json=False)
            else:
                print("Invalid choice.")
        except Exception as e:
            print(f"Error: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aishaman", description="AI Shaman operator CLI")
    p.add_argument("--management-url", default=DEFAULT_MGMT_URL, help="Management API base URL")
    p.add_argument("--json", action="store_true", help="JSON output for automation")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("status", help="Show guardian status (temps/power/queues/lock states)", parents=[common])
    sub.add_parser("health", help="Run health check", parents=[common])
    sub.add_parser("gpus", help="Show GPU ids and ports for selection", parents=[common])

    lock = sub.add_parser("lock", help="Enable maintenance lock (blocks new requests on selected GPU proxy)", parents=[common])
    lock.add_argument("--gpu", type=int, default=None, help="GPU id (omit for all GPUs)")
    lock.add_argument("--reason", default="manual maintenance", help="Lock reason")

    unlock = sub.add_parser("unlock", help="Disable maintenance lock (resume traffic)", parents=[common])
    unlock.add_argument("--gpu", type=int, default=None, help="GPU id (omit for all GPUs)")

    models = sub.add_parser("models", help="Manage backend models directly on each GPU's Ollama backend port", parents=[common])
    models_sub = models.add_subparsers(dest="models_cmd", required=True)
    m_list = models_sub.add_parser("list", help="List models on GPU backend", parents=[common])
    m_list.add_argument("--gpu", type=int, required=False, help="GPU id (omit to list all GPUs)")
    m_pull = models_sub.add_parser("pull", help="Pull model on GPU backend", parents=[common])
    m_pull.add_argument("--gpu", type=int, required=True)
    m_pull.add_argument("--model", required=True)
    m_del = models_sub.add_parser("delete", help="Delete model on GPU backend", parents=[common])
    m_del.add_argument("--gpu", type=int, required=True)
    m_del.add_argument("--model", required=True)

    swap = sub.add_parser("swap", help="Safe swap workflow (lock -> pull -> optional delete old -> probe -> unlock)", parents=[common])
    swap.add_argument("--gpu", type=int, required=True)
    swap.add_argument("--to", required=True, help="Target model")
    swap.add_argument("--from-model", default=None, help="Old model to delete")
    swap.add_argument("--keep-old", action="store_true", help="Do not delete old model")
    swap.add_argument(
        "--pull-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="auto=only pull if missing, always=always pull, never=never pull",
    )
    swap.add_argument("--skip-probe", action="store_true", help="Skip probe step")
    swap.add_argument("--force", action="store_true", help="Bypass swap preflight safety block")
    swap.add_argument("--reason", default="model swap", help="Maintenance lock reason")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = AiShamanClient(args.management_url)

    as_json = bool(getattr(args, "json", False))

    if args.cmd is None:
        return _interactive_menu(client)

    try:
        if args.cmd == "status":
            res = client.status()
        elif args.cmd == "health":
            res = client.health()
        elif args.cmd == "gpus":
            st = client.status()
            res = {"gpus": _gpu_rows(st)}
        elif args.cmd == "lock":
            res = client.lock(args.gpu, args.reason)
        elif args.cmd == "unlock":
            res = client.unlock(args.gpu)
        elif args.cmd == "models":
            st = client.status()
            if args.models_cmd == "list":
                if args.gpu is None:
                    res = _models_list_all(st)
                else:
                    res = _models_list(st, args.gpu)
            elif args.models_cmd == "pull":
                res = _models_pull(st, args.gpu, args.model)
            elif args.models_cmd == "delete":
                res = _models_delete(st, args.gpu, args.model)
            else:
                raise CliError(f"Unsupported models command: {args.models_cmd}")
        elif args.cmd == "swap":
            res = _run_swap(
                client=client,
                gpu_id=args.gpu,
                target_model=args.to,
                old_model=args.from_model,
                keep_old=args.keep_old,
                pull_mode=args.pull_mode,
                skip_probe=args.skip_probe,
                reason=args.reason,
                force=args.force,
            )
        else:
            raise CliError(f"Unsupported command: {args.cmd}")
    except Exception as e:
        err = {"ok": False, "error": str(e)}
        if as_json:
            print(json.dumps(err, indent=2))
        else:
            print(f"ERROR: {e}")
        return 1

    _print_result(res, as_json=as_json)
    if isinstance(res, dict) and res.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
