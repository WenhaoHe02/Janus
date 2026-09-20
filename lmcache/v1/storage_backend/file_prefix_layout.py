# SPDX-License-Identifier: Apache-2.0
"""Validate partial filesystem reads before admitting cache-prefix hits."""

# Standard
from collections.abc import Callable, Sequence
from typing import Any


def prefix_layout_fits(stored: Sequence[int], requested: Sequence[int]) -> bool:
    """Return whether requested groups equal a contiguous prefix of stored bytes.

    Args:
        stored: Logical byte length of each stored group, excluding padding.
        requested: Logical byte length of each requested group in the same order.
    Returns:
        False for missing bytes or shifted nonempty groups. Trailing groups may
        be omitted; an interior group cannot be removed by a flat prefix read.
    """
    if len(stored) != len(requested):
        return False
    stored_offset = requested_offset = 0
    for available, needed in zip(stored, requested, strict=True):
        if available < 0 or needed < 0:
            return False
        if needed and (needed > available or stored_offset != requested_offset):
            return False
        stored_offset += available
        requested_offset += needed
    return True


def filter_file_lookup_prefix(
    chunks: list[tuple[int, int, Any]],
    hit_chunks: int,
    block_mapping: dict[str, list[Any]],
    storage_manager: Any,
    requested_group_bytes: Callable[[int, int], list[int]],
    *,
    pin: bool,
) -> int:
    """Trim filesystem hits before the scheduler commits to absent tail state.

    Args:
        chunks: Ordered logical start/end/cache-key tuples from lookup.
        hit_chunks: Prefix count reported by ordinary backend contains.
        block_mapping: Lookup locations and keys; trimmed in place on a miss.
        storage_manager: Manager exposing storage_backends and batched_unpin.
        requested_group_bytes: Exact retrieve-layout byte lengths for a chunk.
        pin: Whether lookup pinned keys that must be released when trimmed.
    Returns:
        Readable contiguous prefix count. This function does not load, synthesize,
        or zero-fill KV. Missing state must be recomputed by the normal scheduler.
    """
    if hit_chunks <= 0 or "LocalDiskBackend" not in block_mapping:
        return hit_chunks
    backend = storage_manager.storage_backends["LocalDiskBackend"]
    locations = list(block_mapping)
    index = locations.index("LocalDiskBackend")
    preceding = sum(len(block_mapping[name]) for name in locations[:index])
    keys = block_mapping["LocalDiskBackend"]
    readable = 0
    for offset, key in enumerate(keys):
        start, end, expected_key = chunks[preceding + offset]
        if key != expected_key:
            raise ValueError("Filesystem lookup key order differs from logical chunks")
        with backend.disk_lock:
            entry = backend.dict.get(key)
            if entry is None:
                break
            shapes = entry.shapes or [entry.shape]
            dtypes = entry.dtypes or [entry.dtype]
            stored = [int(shape.numel()) * dtype.itemsize
                      for shape, dtype in zip(shapes, dtypes, strict=True)]
        if not prefix_layout_fits(stored, requested_group_bytes(start, end)):
            break
        readable += 1
    if readable == len(keys):
        return hit_chunks
    if pin:
        storage_manager.batched_unpin(keys[readable:], ["LocalDiskBackend"])
        for location in locations[index + 1:]:
            storage_manager.batched_unpin(block_mapping[location], [location])
    if readable:
        block_mapping["LocalDiskBackend"] = keys[:readable]
    else:
        del block_mapping["LocalDiskBackend"]
    for location in locations[index + 1:]:
        del block_mapping[location]
    return preceding + readable
