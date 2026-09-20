# SPDX-License-Identifier: Apache-2.0
"""Compile all production GLM proxy buckets before deployment readiness."""

from __future__ import annotations

# Standard
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HISTORY_LENGTHS = (65_536, 98_304, 196_608, 270_336)
APPEND_LENGTHS = (
    32,
    96,
    192,
    384,
    768,
    1_536,
    3_072,
    6_144,
    12_288,
    24_576,
)


def hit_tokens(container: str, request_id: str, started: float) -> int:
    """Return the exact LMCache hit count logged for one warmup request.

    Args:
        container: Serving container name.
        request_id: OpenAI response identifier.
        started: Wall-clock request start used to bound Docker logs.

    Returns:
        Logged LMCache hit tokens.

    Raises:
        RuntimeError: If the request has no unique matching hit record.
    """
    log = subprocess.check_output(
        ["docker", "logs", "--since", str(started), container],
        stderr=subprocess.STDOUT,
        text=True,
    )
    matches = re.findall(
        rf"Reqid:\s*{re.escape(request_id)}[^\n]*LMCache hit tokens:\s*(\d+)",
        log,
    )
    values = {int(value) for value in matches}
    if len(values) != 1:
        raise RuntimeError(
            f"warmup request {request_id} has ambiguous hit records: {values}"
        )
    return values.pop()


def main() -> int:
    """Seed one prefix, compile the finite Q/K grid, and publish a sentinel."""
    sys.path.insert(0, "/tmp")
    from run_glm52_recovery_smoke import (  # noqa: PLC0415
        stream_completion,
        token_sequence,
        wait_reuse_ready,
    )

    container = os.getenv("LMCACHE_PROXY_WARMUP_CONTAINER", "dsv4-csa-cp8-off")
    result = Path(
        os.getenv(
            "LMCACHE_PROXY_WARMUP_RESULT",
            "/tmp/glm_proxy_bucket_warmup_v438.json",
        )
    )
    result.unlink(missing_ok=True)
    temporary = result.with_suffix(result.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    seed_value = 438_001
    prefix = token_sequence(max(HISTORY_LENGTHS), seed_value)
    # GLM needs a cacheable terminal call when a prefix ends near a scheduler
    # boundary. The extra block is only part of the internal seed request.
    padding = token_sequence(256, seed_value + 1)
    seed_started = time.time()
    seed_tokens = prefix + padding
    seed = stream_completion(seed_tokens, skip_save=False)
    if seed.request_id is None:
        raise RuntimeError("proxy bucket seed returned no request ID")
    seed_hit_tokens = hit_tokens(container, seed.request_id, seed_started)
    if seed_hit_tokens == len(seed_tokens):
        ready: dict[str, object] = {
            "status": "already_seeded",
            "hit_tokens": seed_hit_tokens,
        }
    else:
        ready = wait_reuse_ready(seed.request_id)

    requested_histories = os.getenv("LMCACHE_PROXY_WARMUP_HISTORIES", "")
    histories = (
        tuple(int(value) for value in requested_histories.split(","))
        if requested_histories
        else HISTORY_LENGTHS
    )
    if not histories or any(history not in HISTORY_LENGTHS for history in histories):
        raise ValueError(
            "LMCACHE_PROXY_WARMUP_HISTORIES must select configured histories"
        )

    rows: list[dict[str, object]] = []
    for history in histories:
        for append in APPEND_LENGTHS:
            if history + append + 4 > 278_528:
                continue
            suffix = token_sequence(
                append,
                seed_value + history * 17 + append * 31,
            )
            next_seed_token = (
                prefix[history] if history < len(prefix) else padding[0]
            )
            if suffix and suffix[0] == next_seed_token:
                suffix[0] = 1_000 + ((suffix[0] - 999) % 28_000)
            started = time.time()
            timing = stream_completion(
                prefix[:history] + suffix,
                skip_save=True,
            )
            if timing.request_id is None or timing.ttft_s is None:
                raise RuntimeError("proxy bucket warmup returned no first token")
            restored = hit_tokens(container, timing.request_id, started)
            if restored != history:
                raise RuntimeError(
                    f"proxy bucket warmup hit {restored}, expected {history}"
                )
            row = {
                "history": history,
                "append": append,
                "request_id": timing.request_id,
                "ttft_ms": timing.ttft_s * 1_000.0,
                "hit_tokens": restored,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    container_id = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Id}}", container],
        text=True,
    ).strip()
    payload = {
        "status": "ready",
        "container_id": container_id,
        "query_tile_rows": 8_192,
        "key_bucket": True,
        "histories": list(histories),
        "appends": list(APPEND_LENGTHS),
        "seed": asdict(seed),
        "reuse_ready": ready,
        "requests": rows,
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result)
    print(f"GLM_PROXY_BUCKETS_READY path={result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
