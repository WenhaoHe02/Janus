# SPDX-License-Identifier: Apache-2.0
"""Profile-only append log with short stdout references, never large messages."""

# Standard
import fcntl
import json
import os
from pathlib import Path
import sys
from typing import Any


def append_profile_record(kind: str, record: dict[str, Any]) -> dict[str, Any]:
    """Store one complete record and return an immutable byte-range reference.

    Args:
        kind: Legacy or counted device batches, counted volume or finish
            records, or completed CPU raw writes.
        record: JSON-serializable telemetry including native rank and pid.
    Returns:
        A short reference emitted only after the complete record is flushed.
    Raises:
        ValueError: If the channel is unknown.
        OSError: If the profile file cannot be written.
    """
    if kind not in {
        "tutti_device", "tutti_counted_volume", "cpu_raw_write",
        "tutti_counted_device", "tutti_counted_device_finished",
    }:
        raise ValueError("unknown profile record channel")
    torch = sys.modules.get("torch")
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_initialized():
        native_rank = int(distributed.get_rank())
        if int(record["rank"]) not in {-1, native_rank}:
            raise RuntimeError("environment rank disagrees with initialized worker rank")
        record = {**record, "rank": native_rank}
    directory = Path("/tmp/lmcache_io_records_v318")
    directory.mkdir(exist_ok=True)
    path = directory / f"{kind}_pid{os.getpid()}_rank{int(record['rank'])}.jsonl"
    encoded = (json.dumps(record, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.seek(0, os.SEEK_END)
        offset = output.tell()
        output.write(encoded)
        output.flush()
        fcntl.flock(output, fcntl.LOCK_UN)
    return {
        "kind": kind,
        "path": str(path),
        "offset": offset,
        "length": len(encoded),
        "rank": int(record["rank"]),
        "pid": int(record["pid"]),
    }
