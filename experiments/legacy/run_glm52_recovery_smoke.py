# SPDX-License-Identifier: Apache-2.0
"""Validate the recovered GLM cached-prefix path with one 57K+8K cell."""

from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


@dataclass(frozen=True)
class CompletionTiming:
    """Timing and token metadata for one streaming completion."""

    request_id: str | None
    ttft_s: float | None
    e2e_s: float
    prompt_tokens: int
    completion_tokens: int


def stream_completion(
    prompt: list[int],
    *,
    skip_save: bool,
    endpoint: str = "http://127.0.0.1:8000",
) -> CompletionTiming:
    """Issue one streaming request and measure TTFT.

    Args:
        prompt: Token IDs supplied directly to vLLM.
        skip_save: Whether LMCache must avoid publishing the request suffix.
        endpoint: OpenAI-compatible server base URL.

    Returns:
        Timing and token-count metadata returned by the server.
    """
    payload: dict[str, Any] = {
        "model": "deepseek-v4-pro",
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if skip_save:
        payload["kv_transfer_params"] = {"lmcache.skip_save": True}
    request = urllib.request.Request(
        endpoint + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token: float | None = None
    request_id: str | None = None
    usage: dict[str, Any] = {}
    with urllib.request.urlopen(request, timeout=900) as response:
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
                if choice.get("text") and first_token is None:
                    first_token = time.perf_counter()
    finished = time.perf_counter()
    return CompletionTiming(
        request_id=request_id,
        ttft_s=first_token - started if first_token is not None else None,
        e2e_s=finished - started,
        prompt_tokens=int(usage.get("prompt_tokens") or len(prompt)),
        completion_tokens=int(usage.get("completion_tokens") or 4),
    )


def post_rank(port: int, request_id: str) -> dict[str, Any]:
    """Read one rank's asynchronous save readiness.

    Args:
        port: Rank-local internal API port.
        request_id: Cold request whose generation must be reusable.

    Returns:
        Rank-local readiness response.
    """
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/write_slack/reuse_ready",
        data=json.dumps({"request_id": request_id}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        result = json.loads(response.read().decode())
    if not isinstance(result, dict):
        raise RuntimeError(f"rank {port - 7000} returned a non-object status")
    return result


def wait_reuse_ready(request_id: str) -> dict[str, Any]:
    """Wait until all eight ranks can reuse the cold generation.

    Args:
        request_id: Cold request whose generation must be reusable.

    Returns:
        Barrier duration, poll count, and terminal rank states.

    Raises:
        TimeoutError: If all ranks are not ready within 15 minutes.
    """
    started = time.perf_counter()
    deadline = started + 900
    polls = 0
    statuses: list[dict[str, Any]] = []
    while time.perf_counter() < deadline:
        polls += 1
        try:
            statuses = [post_rank(7000 + rank, request_id) for rank in range(8)]
        except (OSError, urllib.error.URLError):
            statuses = []
        if statuses and all(status.get("state") == "ready" for status in statuses):
            return {
                "wait_s": time.perf_counter() - started,
                "polls": polls,
                "statuses": statuses,
            }
        time.sleep(0.1)
    raise TimeoutError(f"generation {request_id} was not ready on all ranks")


def token_sequence(length: int, seed: int) -> list[int]:
    """Build deterministic, valid-looking non-special token IDs.

    Args:
        length: Number of token IDs to generate.
        seed: Per-request sequence seed.

    Returns:
        A deterministic token-ID list below 30,000.
    """
    return [1000 + ((seed + index * 131) % 28000) for index in range(length)]


def main() -> None:
    """Run one cold write, one warm hit, and three formal 57K+8K hits."""
    prefix_tokens = 57_088
    recompute_tokens = 8_192
    seed = int(time.time()) % 10_000
    base = token_sequence(prefix_tokens, seed)

    cold = stream_completion(base, skip_save=False)
    if cold.request_id is None:
        raise RuntimeError("cold request returned no request ID")
    print(json.dumps({"event": "cold_complete", **asdict(cold)}, sort_keys=True))
    ready = wait_reuse_ready(cold.request_id)
    print(json.dumps({"event": "all_ranks_ready", **ready}, sort_keys=True))
    time.sleep(2)

    hits: list[dict[str, Any]] = []
    for repetition in range(4):
        suffix = token_sequence(recompute_tokens, seed + 10_000 + repetition * 997)
        timing = stream_completion(base + suffix, skip_save=True)
        record = {
            "event": "hit_complete",
            "repetition": repetition,
            "is_warmup": repetition == 0,
            "expected_hit_tokens": prefix_tokens,
            "recompute_tokens": recompute_tokens,
            **asdict(timing),
        }
        hits.append(record)
        print(json.dumps(record, sort_keys=True))
        time.sleep(1)

    result = {
        "configuration": {
            "prefix_tokens": prefix_tokens,
            "recompute_tokens": recompute_tokens,
            "formal_repetitions": 3,
            "profile_enabled": False,
        },
        "cold": asdict(cold),
        "ready": ready,
        "hits": hits,
    }
    output = Path("/home/zbuser02/glm52_recovery_57k_8k_20260824.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "result_saved", "path": str(output)}))


if __name__ == "__main__":
    main()
