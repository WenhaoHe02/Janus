# SPDX-License-Identifier: Apache-2.0
"""Request-local reuse of DSv4 chunk shapes, never of KV contents."""

# Standard
from collections.abc import Callable, Sequence
from typing import TypeVar

Shape = TypeVar("Shape")


def dsv4_chunk_shapes(
    ranges: Sequence[tuple[int, int]],
    *,
    total_tokens: int,
    tail_tokens: int,
    make_shapes: Callable[[int, int], Sequence[Shape]],
    reuse: bool,
) -> list[list[Shape]]:
    """Build independent shape lists, reusing equivalent layouts within this call.

    Args:
        ranges: Ordered non-negative, nonempty token ranges.
        total_tokens: Complete prompt length, including recomputed tokens.
        tail_tokens: Number of trailing tokens retaining tail-only groups.
        make_shapes: Factory whose immutable shape entries depend only on
            chunk length and whether its end overlaps the request tail.
            Layout and feature configuration must stay fixed during this call.
        reuse: If false, call the factory for every range (reference behavior).

    Returns:
        One fresh list per input range. No result survives this call, so new
        requests, layouts, or modes cannot inherit an old template.

    Raises:
        ValueError: If prompt/tail lengths or a chunk range are invalid.
    """
    if total_tokens < 0 or tail_tokens < 0:
        raise ValueError("prompt and tail lengths must be non-negative")
    tail_start = max(0, total_tokens - tail_tokens)
    templates: dict[tuple[int, bool], tuple[Shape, ...]] = {}
    result = []
    for start, end in ranges:
        if not 0 <= start < end <= total_tokens:
            raise ValueError("chunk range must fit the complete prompt")
        key = (end - start, end > tail_start)
        if not reuse or key not in templates:
            templates[key] = tuple(make_shapes(start, end))
        result.append(list(templates[key]))
    return result
