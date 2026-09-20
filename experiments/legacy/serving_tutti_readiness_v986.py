# SPDX-License-Identifier: Apache-2.0
"""Reject Tutti measurements when only the HTTP server/environment is ready."""

# Standard
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any
import urllib.error
import urllib.request


def validate_tutti_readiness(
    arm: str, responses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Require eight successful public loader/planner snapshots for Tutti arms.

    Args:
        arm: k_local, no_prefetch, ssd_only, or dram.
        responses: Per-rank HTTP code and public status response.
    Returns:
        Snapshots ordered by rank; non-Tutti arms return an empty list.
    Raises:
        ValueError: The arm is unknown or ranks are duplicated/missing.
        RuntimeError: Any Tutti loader/planner is unavailable.
    Notes:
        This proves loader/planner presence, not successful request I/O or speed.
        Both still require the request-level evidence recorded by the runner.
    """
    if arm not in ("k_local", "no_prefetch", "ssd_only", "dram"):
        raise ValueError(f"Unsupported arm: {arm}")
    if arm in ("ssd_only", "dram"):
        return []
    if sorted(row["rank"] for row in responses) != list(range(8)):
        raise ValueError("Expected exactly one readiness response for each rank")
    failed = [
        row["rank"]
        for row in responses
        if row["http_status"] != 200 or row.get("body", {}).get("status") != "success"
    ]
    if failed:
        raise RuntimeError(f"Tutti loader/planner unavailable on ranks {failed}")
    return sorted(responses, key=lambda row: row["rank"])


def read_tutti_readiness(arm: str) -> list[dict[str, Any]]:
    """Check existing read-only worker endpoints without initializing any loader.

    Args:
        arm: Selected serving comparison arm.
    Returns:
        Eight validated snapshots for Tutti, empty for SSD-only/DRAM.
    Raises:
        ValueError: Unknown arm or malformed response.
        RuntimeError: Any loader/planner is unavailable or endpoint fails.
    """
    if arm in ("ssd_only", "dram"):
        return []
    if arm not in ("k_local", "no_prefetch"):
        raise ValueError(f"Unsupported arm: {arm}")

    def read_rank(rank: int) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{7000 + rank}/write_slack/status", timeout=5
            ) as response:
                return {
                    "rank": rank,
                    "http_status": response.status,
                    "body": json.load(response),
                }
        except urllib.error.HTTPError as error:
            return {
                "rank": rank,
                "http_status": error.code,
                "body": {"error": error.read().decode(errors="replace")},
            }
        except (OSError, ValueError) as error:
            return {"rank": rank, "http_status": 0, "body": {"error": str(error)}}

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(read_rank, range(8)))
    return validate_tutti_readiness(arm, responses)
