# SPDX-License-Identifier: Apache-2.0
"""Explicit, scoped transition of future0.005 batches to closed-loop serving."""

# Standard
import asyncio
from collections.abc import Awaitable, Callable
import json
from pathlib import Path


def use_serial_arrivals(rate: float, output: Path, policy_file: Path) -> bool:
    """Select serial scheduling only for authorized future0.005 experiment roots.

    Args:
        rate: Requested legacy coordinator slot; not an actual rate in serial mode.
        output: New batch output directory.
        policy_file: Explicit user-authorized policy; absence retains periodic mode.
    Returns:
        Whether to await full completion before dispatching each next request.
    Raises:
        ValueError: Policy does not describe the authorized0.005-only transition.
    """
    if rate != 0.005 or not policy_file.exists():
        return False
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if policy.get("replace_requested_rate") != 0.005 or policy.get("mode") != "serial":
        raise ValueError("Invalid0.005-only serial arrival policy")
    destination = output.resolve()
    return any(
        destination.is_relative_to(Path(root).resolve())
        for root in policy["output_roots"]
    )


async def dispatch_requests(
    rows: list[dict], session: Callable[[int, dict], Awaitable[None]], serial: bool
) -> None:
    """Dispatch rows serially or concurrently without changing their output budgets.

    Args:
        rows: Frozen request rows in source order.
        session: Sends and awaits the complete response for one request.
        serial: Await each response before starting another session if true.
    Returns:
        None after every request has completed; propagates session exceptions.
    """
    if serial:
        for index, row in enumerate(rows):
            await session(index, row)
    else:
        await asyncio.gather(*(session(index, row) for index, row in enumerate(rows)))
