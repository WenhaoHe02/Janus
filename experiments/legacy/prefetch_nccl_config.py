# SPDX-License-Identifier: Apache-2.0
"""Optional, communicator-local NCCL resource limits for speculative KV transfer."""

# Standard
import os
from typing import Any, Mapping


def prefetch_nccl_group_options(
    distributed: Any, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Build fresh NCCL options without mutating global NCCL environment settings.

    Args:
        distributed: Initialized torch.distributed API (injectable for CPU tests).
        environment: Configuration mapping; defaults to process environment.

    Returns:
        Empty kwargs when LMCACHE_SSD_TP_GATHER_MAX_CTAS is zero/unset, preserving
        existing new_group behavior. Otherwise a fresh pg_options object limiting
        only the communicator receiving it to at most the configured CTA count.

    Raises:
        ValueError: Negative/noninteger limits or conflicting legacy environment
            channel control are requested.
        RuntimeError: This torch/NCCL API cannot express the requested limit.

    Notes:
        Model TP/EP groups are not modified. Each new prefetch communicator must
        receive its own options object. This changes scheduling, not KV contents,
        prediction results, collective order or distributed partitioning.
    """
    env = os.environ if environment is None else environment
    limit = int(env.get("LMCACHE_SSD_TP_GATHER_MAX_CTAS", "0"))
    if limit < 0:
        raise ValueError("prefetch NCCL CTA limit must be nonnegative")
    if limit == 0:
        return {}
    if int(env.get("LMCACHE_SSD_TP_GATHER_MAX_NCHANNELS", "0")) != 0:
        raise ValueError(
            "do not combine per-group CTA limit with global channel override"
        )
    factory = getattr(getattr(distributed, "ProcessGroupNCCL", None), "Options", None)
    if factory is None:
        raise RuntimeError("per-communicator NCCL options are unavailable")
    options = factory()
    config = getattr(options, "config", None)
    if config is None or not all(
        hasattr(config, key) for key in ("min_ctas", "max_ctas")
    ):
        raise RuntimeError("this NCCL options API cannot set communicator CTA limits")
    config.min_ctas = 1
    config.max_ctas = limit
    return {"pg_options": options}
