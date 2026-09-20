# SPDX-License-Identifier: Apache-2.0
"""Seed DSv4 cache, then capture only one formal request through first token."""

from __future__ import annotations

# Standard
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

# First Party
from run_glm52_recovery_smoke import stream_completion, wait_reuse_ready


def _token_sequence(length: int, seed: int) -> list[int]:
    """Return deterministic non-special token IDs."""
    return [1000 + ((seed + index * 131) % 28000) for index in range(length)]


def _stream_formal(
    prompt: list[int],
    *,
    output_tokens: int,
    nsys_command: list[str],
    session: str,
) -> dict[str, object]:
    """Stream one formal request and stop Nsys at its first output token."""
    payload = {
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "kv_transfer_params": {"lmcache.skip_save": True},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_perf_ns = time.perf_counter_ns()
    started_wall_ns = time.time_ns()
    first_perf_ns: int | None = None
    first_wall_ns: int | None = None
    stop_started_wall_ns: int | None = None
    stop_finished_wall_ns: int | None = None
    request_id: str | None = None
    output_count = 0
    usage: dict[str, object] = {}
    token_wall_ns: list[int] = []
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            request_id = request_id or chunk.get("id")
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                token_ids = choice.get("token_ids") or []
                text = choice.get("text") or ""
                emitted = len(token_ids) if token_ids else int(bool(text))
                if emitted <= 0:
                    continue
                now_perf_ns = time.perf_counter_ns()
                now_wall_ns = time.time_ns()
                if first_perf_ns is None:
                    first_perf_ns = now_perf_ns
                    first_wall_ns = now_wall_ns
                    stop_started_wall_ns = time.time_ns()
                    subprocess.run(
                        [*nsys_command, "stop", "--session", session],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    stop_finished_wall_ns = time.time_ns()
                output_count += emitted
                token_wall_ns.extend([now_wall_ns] * emitted)
    finished_perf_ns = time.perf_counter_ns()
    finished_wall_ns = time.time_ns()
    if first_perf_ns is None or first_wall_ns is None:
        raise RuntimeError("formal completion returned no output token")
    return {
        "request_id": request_id,
        "prompt_tokens": int(usage.get("prompt_tokens") or len(prompt)),
        "completion_tokens": int(usage.get("completion_tokens") or output_count),
        "observed_output_tokens": output_count,
        "started_perf_ns": started_perf_ns,
        "first_token_perf_ns": first_perf_ns,
        "finished_perf_ns": finished_perf_ns,
        "started_wall_ns": started_wall_ns,
        "first_token_wall_ns": first_wall_ns,
        "finished_wall_ns": finished_wall_ns,
        "nsys_stop_started_wall_ns": stop_started_wall_ns,
        "nsys_stop_finished_wall_ns": stop_finished_wall_ns,
        "ttft_s": (first_perf_ns - started_perf_ns) / 1e9,
        "decode_after_first_s": (finished_perf_ns - first_perf_ns) / 1e9,
        "e2e_s": (finished_perf_ns - started_perf_ns) / 1e9,
        "token_wall_ns_first_512": token_wall_ns[:512],
    }


def main() -> None:
    """Run cold seed, warmup, and one externally controlled Nsys formal."""
    history_tokens = int(sys.argv[1])
    append_tokens = int(sys.argv[2])
    output_tokens = int(sys.argv[3])
    output_path = Path(sys.argv[4])
    seed = int(sys.argv[5])
    session = sys.argv[6]
    trace_base = os.environ["NSYS_TRACE_BASE"]
    nsys = os.environ.get(
        "NSYS_BIN",
        "/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys",
    )
    nsys_command = (
        ["sudo", "-n", nsys]
        if os.environ.get("NSYS_USE_SUDO", "1") == "1"
        else [nsys]
    )
    history = _token_sequence(history_tokens, seed)
    append = _token_sequence(append_tokens, seed + 1_000_000)
    cold = stream_completion(history, skip_save=False)
    if cold.request_id is None:
        raise RuntimeError("cold request returned no request id")
    ready = wait_reuse_ready(cold.request_id)
    time.sleep(2)
    warmup = stream_completion(history + append, skip_save=True)
    time.sleep(2)
    subprocess.run(
        [
            *nsys_command,
            "start",
            "--session",
            session,
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "-o",
            trace_base,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    formal = _stream_formal(
        history + append,
        output_tokens=output_tokens,
        nsys_command=nsys_command,
        session=session,
    )
    result = {
        "configuration": {
            "history_tokens": history_tokens,
            "append_tokens": append_tokens,
            "requested_output_tokens": output_tokens,
            "seed": seed,
            "nsys_session": session,
            "nsys_trace_base": trace_base,
        },
        "cold": asdict(cold),
        "ready": ready,
        "warmup": asdict(warmup),
        "formal": formal,
    }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "profile_complete", **formal}, sort_keys=True))
    print(json.dumps({"event": "result_saved", "path": str(output_path)}))


if __name__ == "__main__":
    main()
