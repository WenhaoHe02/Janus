# SPDX-License-Identifier: Apache-2.0
"""Correct GLM sparse-only restoration flags without enabling prediction."""

# Standard
from collections.abc import Mapping

# Local
from corrective_only_configuration_v1144 import corrective_only_flags as original_flags


def corrective_only_flags(model: str, baseline: Mapping[str, str]) -> dict[str, str]:
    """Enable required compact storage and device correction for GLM only.

    Args:
        model: glm, flash, or pro.
        baseline: Original no-prefetch configuration, left unmodified.
    Returns:
        GLM flags with compact-layout production and GPU miss submission enabled;
        predictive execution and real/proxy Indexer sharding stay disabled.
        Other models retain their existing configuration exactly.
    Raises:
        ValueError: The model or baseline is unsupported.
    """
    flags = original_flags(model, baseline)
    if model == 'glm':
        flags.update(LMCACHE_INDEXER_ENABLE_PREFETCH='1',
                     LMCACHE_GLM_DEVICE_SELECTION_MISS_READ='1')
    return flags
