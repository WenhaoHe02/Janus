# SPDX-License-Identifier: Apache-2.0
"""Prediction-only ablation controls layered on the current DS serving baseline.

This module does not install kernels or start experiments. Exact-K needs the
compatible exact-sync implementation and runtime collective evidence; setting
the flag alone is not proof that the path executed.
"""
# Local
from serving_ds_configuration_v890 import environment as serving_environment


def environment(arm: str) -> dict[str, str]:
    """Return four prediction variants without changing true Indexer or lookahead.

    Args:
        arm: k_local, k_exact_sync, q_sync, or unsharded.
    Returns:
        Current serving controls with only prediction partition/exchange changes.
    Raises:
        ValueError: The requested variant is unsupported.
    """
    variants = {
        'k_local': ('key_sharded_owner', '8', '0', '16', '0'),
        'k_exact_sync': ('key_sharded_owner', '8', '0', '16', '1'),
        'q_sync': ('query', '8', '1', '1', '0'),
        'unsharded': ('query', '1', '0', '1', '0'),
    }
    if arm not in variants:
        raise ValueError('Unsupported prediction ablation arm')
    flags = serving_environment('k_local')
    mode, size, exchange, interleave, exact = variants[arm]
    flags.update({
        'LMCACHE_CSA_PREFETCH_CP_MODE': mode,
        'LMCACHE_CSA_PREFETCH_CP_SIZE': size,
        'LMCACHE_CSA_PREFETCH_CP_EXCHANGE_IDS': exchange,
        'LMCACHE_CSA_PREFETCH_CP_INTERLEAVE': interleave,
        'LMCACHE_CSA_PREFETCH_EXACT_K_SYNC': exact,
    })
    return flags
