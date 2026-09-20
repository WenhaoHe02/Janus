# SPDX-License-Identifier: Apache-2.0
"""Phase classification for opt-in single-request, prefill-first batches."""

# Standard
from typing import Protocol


class RequestProgress(Protocol):
    """Public token counters required to classify unfinished prompt work."""

    num_computed_tokens: int
    num_prompt_tokens: int
    num_output_tokens: int


def is_prompt_work(request: RequestProgress) -> bool:
    """Return whether a request still needs prompt work or its first sample.

    Cached prompts still require their final input token/first sample. Paused
    decode requests retain their counters; no KV release or counter reset occurs.
    """
    return (request.num_computed_tokens < request.num_prompt_tokens
            or request.num_output_tokens == 0)


def phase_accepts(request: RequestProgress, phase: str | None) -> bool:
    """Accept a request in its phase, or preserve legacy scheduling for None.

    Raises:
        ValueError: The phase name is unsupported.
    """
    if phase is None:
        return True
    if phase not in ('prefill', 'decode'):
        raise ValueError(phase)
    return is_prompt_work(request) == (phase == 'prefill')
