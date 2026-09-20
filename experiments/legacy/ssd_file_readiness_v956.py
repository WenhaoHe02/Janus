# SPDX-License-Identifier: Apache-2.0
"""Drain LocalDiskBackend writes before unmeasured serving preparation proceeds."""

# Standard
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import urllib.request


def snapshot_rank(rank: int) -> dict:
    """Read pending puts and published file metadata without modifying a worker.

    Args:
        rank: Local worker rank, with its API listening at 7000 + rank.
    Returns:
        Snapshot of the public disk-worker queue and disk metadata dictionary.
    Raises:
        HTTPError: The worker API or read-only diagnostic fails.
        ValueError: The response is not a dictionary.
    """
    script = (
        "backend = app.state.lmcache_adapter.lmcache_engine."
        "storage_manager.storage_backends['LocalDiskBackend']\n"
        "with backend.disk_worker.put_lock:\n"
        "    tasks = list(backend.disk_worker.put_tasks)\n"
        "with backend.disk_lock:\n"
        "    entries = []\n"
        "    for v in backend.dict.values():\n"
        "        shapes = v.shapes or [v.shape]\n"
        "        dtypes = v.dtypes or [v.dtype]\n"
        "        size = 0\n"
        "        index = 0\n"
        "        for shape in shapes:\n"
        "            part = dtypes[index].itemsize\n"
        "            for dim in shape:\n"
        "                part = part * dim\n"
        "            size = size + part\n"
        "            index = index + 1\n"
        "        entries.append((v.path, size))\n"
        "result = dict(pending=[str(key) for key in tasks], entries=entries)\n"
    )
    boundary = "lmcache-ssd-readonly-status-v956"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="script"; '
        'filename="status.py"\r\nContent-Type: text/plain\r\n\r\n'
        f"{script}\r\n--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{7000 + rank}/run_script",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        state = ast.literal_eval(response.read().decode())
    if not isinstance(state, dict):
        raise ValueError("Invalid LocalDiskBackend readiness response")
    return dict(rank=rank, **state)


def validate_files(states: list[dict]) -> list[dict]:
    """Validate published byte lengths against files on the bind-mounted host.

    Args:
        states: Complete rank snapshots with no outstanding writes.
    Returns:
        Compact per-rank counts and published byte totals.
    Raises:
        ValueError: Rank evidence is incomplete or writes remain outstanding.
        OSError: A published file is absent or shorter than its metadata size.
    """
    if sorted(s["rank"] for s in states) != list(range(8)):
        raise ValueError("SSD readiness requires all eight unique ranks")
    results = []
    for state in states:
        if state["pending"]:
            raise ValueError("Cannot validate files while puts are pending")
        total = 0
        for path, size in state["entries"]:
            actual = Path(path).stat().st_size
            if actual < size:
                raise OSError(f"Published SSD file is short: {path}: {actual} < {size}")
            total += size
        results.append(dict(rank=state["rank"], files=len(state["entries"]), bytes=total))
    return results


def wait_disk_publication(request_id: str | None = None, timeout_s: float = 900) -> dict:
    """Wait outside formal timing for all eight local-disk put queues to drain.

    Args:
        request_id: Completed seed request ID for provenance, or None before clear.
        timeout_s: Maximum time permitted for queued asynchronous file writes.
    Returns:
        File-size-checked barrier evidence; not a model output correctness claim.
    Raises:
        TimeoutError: A queue remains non-empty past the deadline.
        OSError: A published file is shorter than its metadata claims.
    """
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            states = list(pool.map(snapshot_rank, range(8)))
            if not any(s["pending"] for s in states):
                return dict(
                    basis="local_disk_puts_drained_and_published_file_sizes_checked",
                    request_id=request_id,
                    wait_s=time.monotonic() - start,
                    ranks=validate_files(states),
                )
            if time.monotonic() - start >= timeout_s:
                raise TimeoutError("LocalDiskBackend asynchronous puts did not drain")
            time.sleep(1)
