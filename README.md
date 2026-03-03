# AI Shaman

**Ollama GPU Guardian Proxy**

*a'shamon* -- "Guardian" in the Old Tongue (Wheel of Time)

An async reverse proxy that sits between your consumers and Ollama instances, enforcing GPU power budgets, temperature circuit breakers, model affinity, and request queuing. Built to prevent the kind of simultaneous GPU power spikes that cause hard reboots on shared PSU setups.

## The Problem

Two RTX 3090s on a shared PSU. Both GPUs spiking simultaneously under inference load. 15+ hard reboots in a single day -- no OOM kills, no kernel panics, just abrupt power-off deaths.

## The Solution

```
Consumers (apps, agents, scripts)
        |
        v
+-----------------------------+
|  AI Shaman (Python)          |
|  Port 11434 -> Ollama:11444  |  <- GPU 0
|  Port 11435 -> Ollama:11445  |  <- GPU 1
|                              |
|  * Per-GPU request queue     |
|  * Power budget enforcement  |
|  * Temp circuit breaker      |
|  * Model affinity            |
|  * Full request logging      |
|  * /guardian/status endpoint  |
+-----------------------------+
        |
        v
  Ollama GPU 0 (:11444)    Ollama GPU 1 (:11445)
```

**Zero-change deployment:** Ollama moves to internal ports. AI Shaman listens on the original ports. No consumer code changes needed.

## Features

- **Per-GPU semaphore** -- max 1 concurrent inference per GPU (configurable). Requests queue instead of stampeding.
- **Combined power budget** -- if total GPU power draw exceeds threshold (default 350W), new requests wait until power drops.
- **Temperature circuit breaker** -- per-GPU, pauses at 82C, resumes at 75C (configurable hysteresis).
- **Model affinity** -- pin specific models to specific GPUs. Wrong model = loud 403 rejection with routing instructions. Empty/missing model field = rejected (fail-closed).
- **Blocked admin endpoints** -- `/api/pull`, `/api/delete`, `/api/create`, `/api/copy`, `/api/push` are blocked at the proxy to prevent VRAM thrashing.
- **Transparent streaming** -- SSE and NDJSON responses proxied chunk-by-chunk. Consumers see no difference.
- **Clean-output mode (non-stream JSON)** -- strips `reasoning` / `thinking` fields and de-dupes repeated sentence tails.
- **Management API** (localhost-only) -- `/guardian/status`, `/guardian/metrics`, `/guardian/pause`, `/guardian/resume`, `/guardian/maintenance/lock`, `/guardian/maintenance/unlock`
- **Full request logging** -- timestamp, GPU, model, HTTP status, latency, power draw at request time.
- **nvidia-smi monitoring** -- polls every 3s with 10s timeout (won't hang if GPU driver locks up).
- **Cross-platform** -- Linux and Windows (anywhere nvidia-smi and Python 3.10+ exist).

## Quick Start

```bash
# Clone
git clone https://github.com/benolenick/ai-shaman.git
cd ai-shaman

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install aiohttp

# Edit guardian_config.json for your setup (ports, GPUs, models)

# Move Ollama to internal ports (e.g., 11444/11445)
# Then start the guardian on the original ports:
python ollama_guardian.py
```

## Configuration

```json
{
  "gpus": [
    {
      "id": 0,
      "name": "qwen-gpu",
      "listen_port": 11434,
      "backend_port": 11444,
      "max_concurrent": 1,
      "temp_pause_c": 82,
      "temp_resume_c": 75,
      "power_limit_w": 200,
      "allowed_models": ["qwen3:*", "qwen2.5:*"]
    },
    {
      "id": 1,
      "name": "gptoss-gpu",
      "listen_port": 11435,
      "backend_port": 11445,
      "max_concurrent": 1,
      "temp_pause_c": 82,
      "temp_resume_c": 75,
      "power_limit_w": 200,
      "allowed_models": ["gpt-oss:*"]
    }
  ],
  "combined_power_threshold_w": 350,
  "monitor_interval_s": 3,
  "queue_timeout_s": 300,
  "clean_output": true,
  "request_no_think_hint": true,
  "log_file": "/path/to/guardian.log",
  "management_port": 11450
}
```

### Model Affinity Patterns

- `"qwen3:14b"` -- exact match
- `"qwen3:*"` -- any tag (matches `qwen3:14b`, `qwen3:14b-nothinker-16k`, etc.)
- `"qwen3"` -- bare name, matches all tags (same as `qwen3:*`)

If `allowed_models` is empty or omitted, all models are allowed on that GPU.

### Clean Output Controls

- `clean_output` (default `true`): for non-streaming JSON responses, removes backend reasoning/thinking fields and trims repeated sentence loops.
- `request_no_think_hint` (default `true`): adds best-effort hints to reduce thinking output on queued POST endpoints.

Notes:
- Streaming responses are passed through unchanged by design.
- These controls sanitize output shape/noise but do not change model quality itself.

## What Happens on Wrong Model

```json
{
  "error": "MODEL REJECTED BY AI SHAMAN",
  "rejected_model": "gpt-oss:pinned",
  "rejected_on": {
    "gpu_id": 0,
    "gpu_name": "qwen-gpu",
    "port": 11434,
    "allowed_models": ["qwen3:*", "qwen2.5:*"]
  },
  "hint": "Send 'gpt-oss:pinned' requests to port 11435 (gptoss-gpu) instead.",
  "routing_table": [
    {"gpu": "qwen-gpu", "port": 11434, "allowed_models": ["qwen3:*", "qwen2.5:*"]},
    {"gpu": "gptoss-gpu", "port": 11435, "allowed_models": ["gpt-oss:*"]}
  ],
  "message": "REQUEST BLOCKED: Model 'gpt-oss:pinned' is NOT allowed on GPU 0 (qwen-gpu, port 11434). This GPU only serves: ['qwen3:*', 'qwen2.5:*']. Send 'gpt-oss:pinned' requests to port 11435 (gptoss-gpu) instead."
}
```

## Why Model Affinity and Admin Blocking Matter

Ollama has a dangerous default behavior: when a consumer sends a request for a model that isn't currently loaded, Ollama silently **unloads the current model** and loads the requested one. There's no confirmation, no error, no warning -- it just swaps.

On a multi-GPU setup where each GPU is pinned to a specific model, this is catastrophic:

1. **VRAM thrashing** -- a single misrouted request can unload a 14B model that took 30 seconds to load, replacing it with a model that belongs on a different GPU.
2. **Power spikes** -- loading a model is one of the most power-intensive GPU operations. An uncontrolled model swap on both GPUs simultaneously is exactly the kind of power spike that causes hard reboots.
3. **Silent failure** -- the consumer gets a valid response (from the wrong GPU), so nobody notices the model was swapped until other requests start failing or performance degrades.

AI Shaman prevents this in two ways:

- **Model affinity** rejects requests for the wrong model *before they reach Ollama*, with a loud 403 that tells the consumer exactly which port to use instead.
- **Blocked admin endpoints** (`/api/pull`, `/api/delete`, `/api/create`, `/api/copy`, `/api/push`) prevent consumers from triggering model downloads or deletions through the proxy. To manage models, connect directly to the internal Ollama ports (e.g., `curl http://localhost:11444/api/pull`).

## Systemd Service

```ini
[Unit]
Description=AI Shaman - Ollama GPU Guardian
After=ollama.service ollama-gpu1.service
Wants=ollama.service ollama-gpu1.service

[Service]
Type=simple
User=om
Group=om
WorkingDirectory=/home/om/ollama-guardian
ExecStart=/home/om/ollama-guardian/.venv/bin/python /home/om/ollama-guardian/ollama_guardian.py
Restart=always
RestartSec=5
Environment="GUARDIAN_CONFIG=/home/om/ollama-guardian/guardian_config.json"

[Install]
WantedBy=multi-user.target
```

## Management API

All endpoints on `127.0.0.1:11450` (localhost only):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/guardian/status` | GET | GPU temps, power, VRAM, queue depths, circuit breaker state |
| `/guardian/metrics` | GET | Request counts, latency histograms, power history |
| `/guardian/pause` | POST | Pause a GPU or all GPUs (`{"gpu_id": 0}` or empty for all) |
| `/guardian/resume` | POST | Resume a paused GPU |
| `/guardian/maintenance/lock` | POST | Hard-lock a GPU proxy port (or all) during model swap (`{"gpu_id": 1, "reason": "swap"}`) |
| `/guardian/maintenance/unlock` | POST | Remove maintenance lock from a GPU proxy port (or all) |

## Operator CLI (`aishaman`)

`aishaman` supports both guided human use and automation-safe agent workflows.

Human interactive mode:

```bash
aishaman
```

Agent/script mode:

```bash
aishaman status --json
aishaman lock --gpu 1 --reason "model swap" --json
aishaman swap --gpu 1 --to qwen3.5:latest --from-model gpt-oss:latest --json
aishaman unlock --gpu 1 --json
```

The `swap` workflow performs:
1. Maintenance lock
2. Optional pull target model on backend
3. Optional delete old model on backend
4. Optional probe call
5. Unlock

If a step fails, it attempts automatic unlock.

## Rollback

```bash
bash rollback.sh
```

Stops the guardian, reverts Ollama to original ports, restarts Ollama services. One command, 30 seconds.

## Dependencies

- Python 3.10+
- `aiohttp`
- `nvidia-smi` (comes with NVIDIA drivers)

## License

MIT
