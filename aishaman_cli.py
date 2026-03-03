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
from typing import Any


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


def _run_swap(
    client: AiShamanClient,
    gpu_id: int,
    target_model: str,
    old_model: str | None,
    keep_old: bool,
    skip_pull: bool,
    skip_probe: bool,
    reason: str,
) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    locked = False
    try:
        timeline.append({"step": "lock", "status": "start"})
        lock_res = client.lock(gpu_id, reason)
        timeline.append({"step": "lock", "status": "ok", "result": lock_res})
        locked = True
        time.sleep(0.3)

        status = client.status()

        if not skip_pull:
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
        print("1. View status")
        print("2. Health check")
        print("3. Lock GPU")
        print("4. Unlock GPU")
        print("5. List models on GPU backend")
        print("6. Guided swap workflow")
        print("7. Exit")
        choice = input("Choose [1-7]: ").strip()
        try:
            if choice == "1":
                _print_result(client.status(), as_json=False)
            elif choice == "2":
                _print_result(client.health(), as_json=False)
            elif choice == "3":
                raw = input("GPU id (blank for all): ").strip()
                gpu_id = int(raw) if raw else None
                reason = input("Reason [manual maintenance]: ").strip() or "manual maintenance"
                _print_result(client.lock(gpu_id, reason), as_json=False)
            elif choice == "4":
                raw = input("GPU id (blank for all): ").strip()
                gpu_id = int(raw) if raw else None
                _print_result(client.unlock(gpu_id), as_json=False)
            elif choice == "5":
                gpu_id = int(input("GPU id: ").strip())
                st = client.status()
                _print_result(_models_list(st, gpu_id), as_json=False)
            elif choice == "6":
                gpu_id = int(input("GPU id: ").strip())
                target = input("Target model (e.g. qwen3.5:latest): ").strip()
                old = input("Old model to delete (optional): ").strip() or None
                keep_old = input("Keep old model? [y/N]: ").strip().lower() in {"y", "yes"}
                skip_pull = input("Skip pull step? [y/N]: ").strip().lower() in {"y", "yes"}
                skip_probe = input("Skip probe step? [y/N]: ").strip().lower() in {"y", "yes"}
                reason = input("Lock reason [guided swap]: ").strip() or "guided swap"
                res = _run_swap(client, gpu_id, target, old, keep_old, skip_pull, skip_probe, reason)
                _print_result(res, as_json=False)
            elif choice == "7":
                return 0
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

    sub.add_parser("status", help="Show guardian status", parents=[common])
    sub.add_parser("health", help="Run health check", parents=[common])

    lock = sub.add_parser("lock", help="Enable maintenance lock", parents=[common])
    lock.add_argument("--gpu", type=int, default=None, help="GPU id (omit for all)")
    lock.add_argument("--reason", default="manual maintenance", help="Lock reason")

    unlock = sub.add_parser("unlock", help="Disable maintenance lock", parents=[common])
    unlock.add_argument("--gpu", type=int, default=None, help="GPU id (omit for all)")

    models = sub.add_parser("models", help="Manage backend models directly", parents=[common])
    models_sub = models.add_subparsers(dest="models_cmd", required=True)
    m_list = models_sub.add_parser("list", help="List models on GPU backend", parents=[common])
    m_list.add_argument("--gpu", type=int, required=True)
    m_pull = models_sub.add_parser("pull", help="Pull model on GPU backend", parents=[common])
    m_pull.add_argument("--gpu", type=int, required=True)
    m_pull.add_argument("--model", required=True)
    m_del = models_sub.add_parser("delete", help="Delete model on GPU backend", parents=[common])
    m_del.add_argument("--gpu", type=int, required=True)
    m_del.add_argument("--model", required=True)

    swap = sub.add_parser("swap", help="Safe swap workflow (lock -> pull -> probe -> unlock)", parents=[common])
    swap.add_argument("--gpu", type=int, required=True)
    swap.add_argument("--to", required=True, help="Target model")
    swap.add_argument("--from-model", default=None, help="Old model to delete")
    swap.add_argument("--keep-old", action="store_true", help="Do not delete old model")
    swap.add_argument("--skip-pull", action="store_true", help="Skip pull step")
    swap.add_argument("--skip-probe", action="store_true", help="Skip probe step")
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
        elif args.cmd == "lock":
            res = client.lock(args.gpu, args.reason)
        elif args.cmd == "unlock":
            res = client.unlock(args.gpu)
        elif args.cmd == "models":
            st = client.status()
            if args.models_cmd == "list":
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
                skip_pull=args.skip_pull,
                skip_probe=args.skip_probe,
                reason=args.reason,
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
