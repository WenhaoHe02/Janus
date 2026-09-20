# SPDX-License-Identifier: Apache-2.0
"""Trace-serving arrival schedules and client-observed streaming latencies."""

# Standard
from dataclasses import dataclass, field
import math
import random
from typing import Any


def session_arrivals(count: int, rate: float, seed: int) -> list[float]:
    """Return reproducible Poisson session arrival offsets in seconds.

    Args:
        count: Number of distinct sessions, not a request completion target.
        rate: Offered sessions per second, identical across compared arms.
        seed: Fixed schedule seed; independent of response completion times.
    Returns:
        Absolute offsets from a shared start clock, preserving offered load.
    Raises:
        ValueError: Invalid count or nonpositive/nonfinite rate.
    """
    if count < 0 or not math.isfinite(rate) or rate <= 0:
        raise ValueError('invalid offered session load')
    generator = random.Random(seed)
    elapsed = 0.0
    arrivals = []
    for _ in range(count):
        elapsed += generator.expovariate(rate)
        arrivals.append(elapsed)
    return arrivals


def next_turn_due(response_end: float, tool_delay: float) -> float:
    """Return dependent-turn eligibility after a finite nonnegative tool wait."""
    if not math.isfinite(response_end) or not math.isfinite(tool_delay) or tool_delay < 0:
        raise ValueError('invalid response clock or tool delay')
    return response_end + tool_delay


@dataclass
class StreamObservation:
    """Track application-visible output boundaries without extra model profiling.

    Empty text/usage/finish chunks are not generated-token timestamps. TPOT uses
    source output count and first/last visible chunks, so chunking resolution is
    explicit. No extra logprobs or full prompt-token echo is requested.
    """

    sent: float
    first: float | None = None
    last: float | None = None
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    visible_events: int = 0
    visible_times: list[float] = field(default_factory=list)
    done: bool = False
    finish_reason: str | None = None

    def observe(self, chunk: dict[str, Any], received: float) -> None:
        """Record a parsed SSE payload at its receive clock; raise on SSE errors."""
        if received < self.sent or (self.last is not None and received < self.last):
            raise ValueError('nonmonotonic client clock')
        if chunk.get('error'):
            raise RuntimeError(f"SSE error: {chunk['error']}")
        identity = chunk.get('id')
        if self.request_id and identity and self.request_id != identity:
            raise ValueError('response identity changed within stream')
        self.request_id = self.request_id or identity
        self.usage = chunk.get('usage') or self.usage
        for choice in chunk.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('serving measurement requires one completion')
            self.finish_reason = choice.get('finish_reason') or self.finish_reason
            if choice.get('text'):
                self.first = received if self.first is None else self.first
                self.last = received
                self.visible_events += 1
                self.visible_times.append(received)

    def result(self, ended: float, expected_output: int) -> dict[str, Any]:
        """Validate complete response/O and return metrics; absent boundaries stay null."""
        if not self.done or not self.request_id or self.usage.get('completion_tokens') != expected_output:
            raise ValueError('missing DONE, identity, or original output budget')
        if ended < self.sent or (self.last is not None and ended < self.last):
            raise ValueError('response end precedes output')
        # A single visible chunk carrying multiple tokens has no within-chunk
        # timing resolution. Report TPOT as unavailable, never as zero.
        tpot = None
        if expected_output > 1 and self.visible_events > 1:
            assert self.first is not None and self.last is not None
            tpot = 1000 * (self.last - self.first) / (expected_output - 1)
        return dict(request_id=self.request_id, usage=self.usage,
            send_s=self.sent, first_visible_s=self.first, last_visible_s=self.last,
            end_s=ended, visible_chunks=self.visible_events, finish_reason=self.finish_reason,
            visible_chunk_offsets_ms=[1000*(value-self.sent) for value in self.visible_times],
            ttft_ms=None if self.first is None else 1000 * (self.first-self.sent),
            tpot_ms=tpot, e2e_ms=1000*(ended-self.sent),
            timing_basis='client-visible text chunks; not device/token-exact timing')
