# SPDX-License-Identifier: Apache-2.0
"""Bound native Indexer-K read groups without changing selected/history rows."""


def native_indexer_group_rows(
    rows: int, *, aligned_tail_supported: bool, preferred_rows: int = 0
) -> int:
    """Select a transport grouping for 8448-byte native indexer cache rows.

    Args:
        rows: Exact number of valid native cache rows, excluding append padding.
        aligned_tail_supported: True only when the installed reader copies exact
            tail bytes from sector-aligned DMA through its normal completion gate.
        preferred_rows: Opt-in group width, 2, 4, or 6; zero preserves the existing
            largest-divisor policy (including aligned two-row fallback).
    Returns:
        Number of entries per command group, or zero for the original fallback
        reader. A group is at most 50688 bytes, within the existing 64KiB limit.
    Raises:
        ValueError: Nonpositive row count or unsupported preferred width.

    Notes:
        A short final group does not authorize copying padding into the cache.
        Without the exact-tail reader, only complete groups may use this path.
    """
    if rows <= 0 or preferred_rows not in (0, 2, 4, 6):
        raise ValueError("positive rows and preferred width 0/2/4/6 required")
    if preferred_rows and (aligned_tail_supported or rows % preferred_rows == 0):
        return preferred_rows
    width = next((size for size in (6, 4, 2) if rows % size == 0), 0)
    return width or (2 if aligned_tail_supported else 0)
