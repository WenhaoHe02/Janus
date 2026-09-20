# SPDX-License-Identifier: Apache-2.0
"""Fail-closed preparation checks; publication tickets are not cache lookups."""

from __future__ import annotations

# Standard
from typing import Any


def stored_prefix_tokens(prompt_tokens: int, chunk_size: int) -> int:
    """Return the full-chunk prefix, without changing the actual request length.

    Args:
        prompt_tokens: Unmodified input token count.
        chunk_size: Storage chunk size verified from the deployed configuration.
    Returns:
        Number of input tokens belonging to complete storage chunks.
    Raises:
        ValueError: A count is invalid.
    """
    if prompt_tokens < 0 or chunk_size <= 0:
        raise ValueError("invalid prompt length or storage chunk size")
    return prompt_tokens // chunk_size * chunk_size


def publication_decision(
    states: list[dict[str, Any]],
    *,
    prompt_tokens: int,
    chunk_size: int,
    restored_tokens: int,
    expected_ranks: int = 8,
) -> str:
    """Classify exact-rank evidence after a successfully completed seed request.

    Args:
        states: One request-scoped publication response per rank, in rank order.
        prompt_tokens: Original input length, never padded or truncated.
        chunk_size: Verified storage chunk size.
        restored_tokens: Actual LMCache hit count for this exact request ID.
        expected_ranks: Number of ranks that must respond.
    Returns:
        ``published``, ``already_cached_complete_blocks``, or ``pending``.
        Cached-block evidence does not assert output correctness or certify any
        other request, container, or tail-state format.
    Raises:
        ValueError: Evidence is missing, impossible, or contains failed writes.
    """
    if expected_ranks <= 0 or len(states) != expected_ranks:
        raise ValueError("incomplete rank evidence")
    if not 0 <= restored_tokens <= prompt_tokens:
        raise ValueError("invalid restored-token evidence")
    required = stored_prefix_tokens(prompt_tokens, chunk_size)
    if any(state.get("state") == "failed" for state in states):
        raise ValueError("publication failed on at least one rank")
    if all(state.get("state") == "ready" for state in states):
        if all(int(state.get("token_count", 0)) >= required for state in states):
            return "published"
        raise ValueError("ready ticket does not cover the required prefix")
    if all(state.get("state") == "not_found" for state in states):
        if required > 0 and restored_tokens >= required:
            return "already_cached_complete_blocks"
    return "pending"


def validate_predecessor_evidence(
    records: list[dict[str, Any]],
    *,
    expected_source_ids: list[str],
    container_id: str,
    required_env: dict[str, str],
) -> list[str]:
    """Validate every source against same-container, same-arm observations.

    Args:
        records: Completed per-source observations, not an asserted ID list.
        expected_source_ids: Frozen manifest predecessor IDs.
        container_id: Current serving container ID.
        required_env: Exact controls verified against every current worker.
    Returns:
        Verified predecessor IDs in manifest order.
    Raises:
        ValueError: A source is missing, duplicated, cross-container, wrong-arm,
            or lacks successful request and readiness evidence.
    """
    if len(set(expected_source_ids)) != len(expected_source_ids):
        raise ValueError("duplicate manifest source IDs")
    by_source: dict[str, dict[str, Any]] = {}
    for record in records:
        source_id = record["source_request_id"]
        if source_id in by_source:
            raise ValueError("duplicate predecessor evidence")
        if record.get("container_id") != container_id:
            raise ValueError("cross-container predecessor evidence")
        if record.get("required_worker_env") != required_env:
            raise ValueError("wrong-arm predecessor evidence")
        if not record.get("request_id") or record.get("completion_tokens") != 4:
            raise ValueError("missing successful completion evidence")
        if record.get("readiness") not in (
            "published", "already_cached_complete_blocks"
        ):
            raise ValueError("unverified predecessor readiness")
        by_source[source_id] = record
    if set(by_source) != set(expected_source_ids):
        raise ValueError("predecessor coverage differs from the manifest")
    return list(expected_source_ids)


def validate_warmup_records(
    records: list[dict[str, Any]],
    *, shapes: list[tuple[int, int]], repeats: int,
    current_container_request_ids: set[str],
) -> None:
    """Verify complete shape coverage by successful IDs in this container's log.

    Args:
        records: Previously recorded shape warmups; not publication evidence.
        shapes: Exact history/append pairs required by the frozen manifest.
        repeats: Number of excluded warmup passes per shape.
        current_container_request_ids: IDs observed in current-container logs.
    Returns:
        None when every shape/pass has valid evidence.
    Raises:
        ValueError: A warmup is missing, duplicated, unsuccessful or foreign.
    """
    expected = {(h, a, r) for h, a in shapes for r in range(1, repeats + 1)}
    seen = set()
    for record in records:
        if record.get('phase') != 'shape_warmup_excluded':
            continue
        key = (record['H'], record['A'], record['repeat'])
        completion = record.get('completion', {})
        if key in seen or key not in expected:
            raise ValueError('duplicate or unexpected warmup shape/pass')
        if (completion.get('request_id') not in current_container_request_ids
                or completion.get('completion_tokens') != 4
                or completion.get('prompt_tokens') != record['H'] + record['A']
                or record.get('actual_hit') != record['H']):
            raise ValueError('warmup lacks exact current-container hit evidence')
        seen.add(key)
    if seen != expected:
        raise ValueError('incomplete compiled-shape coverage')
