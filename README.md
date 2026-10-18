# Janus

**SSD-backed KV-cache management for long-context LLM serving.**

**Paper:** [Efficient Agentic LLM Serving over SSD-based Sparse KV Storage](https://arxiv.org/abs/2609.36938)

Janus integrates sparse KV prediction and GPU-direct NVMe I/O with vLLM. It
partitions prediction across tensor-parallel ranks, prefetches sparse KV blocks
ahead of attention, and loads missing blocks on demand.

- **Parallel prediction:** shard the prediction key space across GPUs.
- **Overlapped recovery:** overlap sparse KV reads with model execution.
- **Unified serving:** one launcher for DeepSeek-V4-Flash, DeepSeek-V4-Pro, and GLM-5.2.
- **Request-aware writeback:** throttle background writes during request processing
  to reduce I/O interference, and write without throttling when idle.

[Installation](#installation) · [Quick Start](#quick-start) ·
[Usage](#usage) · [Project Structure](#project-structure) · [License](#license)

## Installation

### Dependencies

- Linux with NVIDIA Hopper GPUs (`sm_90`) and NVMe storage.
- Python 3.12, PyTorch 2.11.0 with CUDA 13.0, and the CUDA 13.0 toolkit.
- vLLM 0.26.0 at revision `ffd46bfab2128bb84146050e98b51a617c6575ab`.
- [Tutti](https://github.com/xPU-IO/Tutti) at revision
  `872b035f8004c6af7aaafda1955410fc964b561e`, with [TUTTI_LOCAL.patch](TUTTI_LOCAL.patch)
  applied before building the driver.

Install the matching serving environment and Tutti backend, then run the following
commands from the repository root:

```bash
python -m pip install -r requirements/build.txt
python -m pip install . --no-build-isolation
```

Janus retains the `lmcache` Python package name for compatibility with the vLLM
connector. Model-specific vLLM overlays are applied by the launcher.

## Quick Start

### 1. Configure the model and storage

```bash
mkdir -p /deployment
cp config/deployment.example.json /deployment/deployment.json
cp config/storage.example.yaml /deployment/storage.yaml
```

Edit the model checkpoint, warmup directory, memory budget, and storage paths.
Storage paths are absolute and ordered by tensor-parallel rank, with one entry
per rank. Provision the KV and indexer regions through the Tutti backend before
starting Janus. Use dedicated cache regions: startup preparation clears them.

| Model | Configuration value |
| --- | --- |
| DeepSeek-V4-Flash | `flash` |
| DeepSeek-V4-Pro | `pro` |
| GLM-5.2 | `glm` |

Predictive prefetching is enabled in the provided deployment template.

### 2. Prepare warmup inputs

The directory specified by `warmup_inputs` contains `tokens.i32le` and
`manifest.json`. Tokenize your inputs with the model's tokenizer and store token
IDs as little-endian int32 values without a file header.

A manifest describes the prompt slices and their reusable prefix lengths:

```json
{
  "model": "janus-model",
  "selected_count": 1,
  "chains": [{
    "first_timed_index": 0,
    "rounds": [{
      "source_request_id": "request-1",
      "token_offset": 0,
      "prompt_tokens": 65792,
      "reusable_input_prefix_tokens": 65536
    }]
  }]
}
```

Offsets and lengths are in tokens. Add entries for the input shapes you plan to
serve and update `selected_count`. The `model` field defines the served API name.

### 3. Start the server

```bash
python tools/submission/prepare_deployment.py \
  --source "$PWD" \
  --deployment /deployment/deployment.json \
  --output /deployment/flash

bash start.sh --model flash \
  --plan /deployment/flash/plan.json \
  --output /deployment/flash-run
```

The launcher builds the native operators, starts the model, and warms up the
configured inputs. Once preparation completes, it writes
`/deployment/flash-run/READY.json`. Keep the launcher running and route requests
after this file appears.

Use fresh output directories for each deployment. To select another model, change
both the configuration's `model` value and the launcher's `--model` argument.
Set `PYTHON` to choose an interpreter other than `python3`.

## Usage

Janus exposes an OpenAI-compatible API at `http://127.0.0.1:8000` by default:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "janus-model",
    "prompt": "Explain how KV caching accelerates long-context inference.",
    "max_tokens": 64,
    "temperature": 0
  }'
```

The run directory contains build, server, warmup, and verification logs.
Stop the foreground launcher with Ctrl+C.

## Project Structure

| Path | Description |
| --- | --- |
| `start.sh` | Unified serving launcher |
| `config/` | Deployment templates and model profiles |
| `lmcache/`, `csrc/` | KV-cache runtime and native operators |
| `overlays/` | Model-specific integrations |
| `experiments/native/` | GPU-driven I/O extension sources |
| `tools/submission/` | Build, configuration, and startup utilities |

## Tests

```bash
python -B -m unittest discover -s tools/submission -p 'test_*.py' -v
```

## License

This repository retains the LMCache [Apache-2.0 license](LICENSE). Third-party
components retain their respective licenses.

Janus builds on [LMCache](https://github.com/LMCache/LMCache),
[vLLM](https://github.com/vllm-project/vllm), and
[Tutti](https://github.com/xPU-IO/Tutti).
