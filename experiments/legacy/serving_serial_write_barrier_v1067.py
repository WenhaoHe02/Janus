# SPDX-License-Identifier: Apache-2.0
"""Serial request boundary: response followed by verified write publication."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Callable


def serial_write_barrier_enabled(serial: bool, configuration_module: str) -> bool:
    """Enable DS serial drain, plus explicitly opted-in GLM serial repeats.

    Periodic and existing GLM runs remain unchanged. GLM opt-in is inherited
    from the new repeat coordinator, never set in an already running client.
    """
    return serial and (
        configuration_module == "serving_ds_configuration_v890"
        or (configuration_module == "serving_glm_configuration_v766"
            and os.getenv("LMCACHE_GLM_SERIAL_WRITE_BARRIER_V1123") == "1")
    )


def wait_serial_write_boundary(
    arm: str, container: str, result: dict, prompt_tokens: int, started_wall: float,
    *, publication: Callable, disk_publication: Callable, rank_status: Callable,
    timeout_s: float = 900,
) -> dict:
    """Wait after a successful response before allowing the next serial request.

    Args:
        arm: Measured backend path.
        container: Exact running container identity.
        result: Completed request including status and request_id.
        prompt_tokens: Unmodified input size.
        started_wall: Actual request start in wall-clock seconds.
        publication: Existing request-specific eight-rank publication barrier.
        disk_publication: Existing ordinary file-write drain and size check.
        rank_status: Read-only GET callable for worker public status endpoints.
        timeout_s: Maximum additional physical-writer drain time.
    Returns:
        Separately timed evidence, excluded from request TTFT/TPOT/E2E but
        included in the overall replay wall time.
    Raises:
        RuntimeError: A request failed or worker state is invalid.
        TimeoutError: Physical writers do not drain.
        ValueError: Unknown arm or absent request identity.
    """
    if result.get("status") != "success":
        raise RuntimeError("Failed serial response; retain raw record and stop")
    request_id = result.get("request_id")
    if not request_id:
        raise ValueError("Serial publication requires an actual request ID")
    began = time.monotonic()
    evidence = {"request_id": request_id, "started_s": began, "arm": arm}
    if arm in ("k_local", "no_prefetch"):
        evidence["publication"] = publication(
            container, request_id, prompt_tokens, started_wall, chunk_size=256
        )
        # Exact generation publication comes FIRST. Empty writer queues alone
        # would miss sidecar preparation that has not submitted its writes yet.
        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                states = list(pool.map(
                    rank_status,
                    [f"http://127.0.0.1:{7000+r}/write_slack/status" for r in range(8)],
                ))
                if any(s.get("status") != "success" for s in states):
                    raise RuntimeError("Missing eight-rank physical writer evidence")
                if not any(s.get(k, 0) for s in states for k in
                           ("queued_waves", "active_waves", "active_io_blocks")):
                    evidence["writer_states"] = states
                    break
                if time.monotonic() - began >= timeout_s:
                    raise TimeoutError("Serial physical writers did not drain")
                time.sleep(0.05)
        evidence["basis"] = "request_publication_then_eight_rank_writer_drain"
    elif arm == "ssd_only":
        evidence["publication"] = disk_publication(request_id)
        evidence["basis"] = "eight_rank_local_disk_publication"
    elif arm == "dram":
        # LocalCPUBackend.batched_submit_put_task is synchronous; it has no
        # asynchronous SSD writer. Do not pretend to measure a disk barrier.
        evidence["basis"] = "synchronous_LocalCPUBackend_put_no_SSD_writer"
    else:
        raise ValueError(f"Unknown arm: {arm}")
    evidence["finished_s"] = time.monotonic()
    evidence["wait_ms"] = (evidence["finished_s"] - began) * 1000
    return evidence
