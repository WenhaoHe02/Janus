# SPDX-License-Identifier: Apache-2.0
"""Request-local scheduling of owner gathers at fixed foreground boundaries."""

# Standard
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
import math
import threading


@dataclass(frozen=True)
class GatherGroup:
    """One unchanged prediction's consumer layers and fixed launch boundary."""

    source: int
    target: int
    consumers: tuple[int, ...]
    boundaries: tuple[int, ...]


class FixedBoundaryGathers:
    """Move transport launch without changing prediction or consumer ordering.

    Full consumers launch after ``source+1`` on the registering foreground thread.
    Staged Shared consumers retain their original release boundaries. The callback
    may wait for local CPU preparation; readiness does not
    decide different launch layers on different ranks. It must enqueue the
    original gather and preserve the normal target-layer completion event.
    """

    def __init__(self) -> None:
        self.request: Hashable | None = None
        self.groups: dict[int, GatherGroup] = {}
        self.launched_layers: list[int] = []
        self.owner_thread: int | None = None
        self.failed = False

    def reset(self, request: Hashable | None = None) -> None:
        """Drop scheduling metadata only; the caller still drains original I/O."""
        self.request = request
        self.groups.clear()
        self.launched_layers.clear()
        self.owner_thread = None
        self.failed = False

    def add(
        self,
        request: Hashable,
        source: int,
        target: int,
        consumers: Sequence[int],
        *,
        staged: bool = False,
    ) -> None:
        """Register a group after its original prediction has been submitted.

        A new request resets metadata. Distance-one bootstrap groups retain
        their original target gate. Malformed or duplicate groups raise
        ValueError; a background-thread registration raises RuntimeError. When
        staged=True, each Shared consumer launches at the release layer specified
        by GLM's evenly spaced staging contract, not at the Full launch boundary.
        """
        layers = tuple(consumers)
        if source < 0 or source >= target or not layers:
            raise ValueError("invalid prediction group")
        if layers[0] != target or tuple(sorted(set(layers))) != layers:
            raise ValueError("consumer layers must be unique and ordered")
        if self.request != request:
            self.reset(request)
        self._check_thread()
        if source + 1 >= target:
            return
        if target in self.groups:
            raise ValueError("prediction group already scheduled")
        boundaries = tuple(
            min(
                target - 1,
                max(
                    source + 1,
                    source + math.ceil((target - source) * ordinal / len(layers)),
                ),
            )
            if staged
            else source + 1
            for ordinal in range(len(layers))
        )
        self.groups[target] = GatherGroup(source, target, layers, boundaries)

    def discard(self, target: int) -> None:
        """Leave a group to its unchanged target gate if it was not launched early."""
        self.groups.pop(target, None)

    def advance(
        self, request: Hashable, completed_layer: int, launch: Callable[[int], bool]
    ) -> tuple[int, ...]:
        """Launch due groups in order, or fail closed on a callback failure.

        Args:
            request: Active request-generation token.
            completed_layer: Foreground layer that just finished its hook.
            launch: Existing per-layer gather API, retaining consumer event waits.
        Returns:
            Consumer layer IDs launched by this call, possibly empty.
        Raises:
            RuntimeError: Stale request, different thread, missed boundary, or
                failed transport submission. A failure is never retried here.
        """
        if not self.groups:
            return ()
        if request != self.request:
            raise RuntimeError("stale fixed-boundary request")
        self._check_thread()
        launched = []
        for target, group in sorted(tuple(self.groups.items())):
            boundary = group.boundaries[0]
            if completed_layer < boundary:
                break
            if completed_layer > boundary:
                raise RuntimeError("fixed foreground launch boundary was skipped")
            try:
                consumed = 0
                for layer, boundary in zip(
                    group.consumers, group.boundaries, strict=True
                ):
                    if boundary > completed_layer:
                        break
                    if not launch(layer):
                        raise RuntimeError(
                            f"owner gather submission failed at L{layer}"
                        )
                    launched.append(layer)
                    self.launched_layers.append(layer)
                    consumed += 1
            except BaseException:
                self.failed = True
                raise
            if consumed == len(group.consumers):
                del self.groups[target]
            else:
                self.groups[target] = GatherGroup(
                    group.source,
                    group.target,
                    group.consumers[consumed:],
                    group.boundaries[consumed:],
                )
        return tuple(launched)

    def _check_thread(self) -> None:
        """Reject dynamic/background collective enqueue and post-failure retries."""
        if self.failed:
            raise RuntimeError("fixed-boundary scheduler is failed")
        current = threading.get_ident()
        if self.owner_thread is None:
            self.owner_thread = current
        elif current != self.owner_thread:
            raise RuntimeError("owner gathers require the original foreground thread")
