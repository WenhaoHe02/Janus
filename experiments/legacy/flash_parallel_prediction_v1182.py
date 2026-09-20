# SPDX-License-Identifier: Apache-2.0
"""Two-arm prediction parallelism ablation on the current DS serving path."""

# Local
from ds_prediction_ablation_configuration_v1128 import environment as variant_environment


def environment(arm: str) -> dict[str, str]:
    """Return K-parallel or unsharded prediction flags, never no-prefetch.

    Args:
        arm: parallel or unsharded.
    Returns:
        Current serving flags with true Indexer CP disabled and lookahead two.
    Raises:
        ValueError: An unsupported ablation arm was requested.
    """
    if arm not in ('parallel', 'unsharded'):
        raise ValueError('Parallel prediction ablation has exactly two arms')
    flags = variant_environment('k_local' if arm == 'parallel' else 'unsharded')
    flags.update(LMCACHE_CORRECTIVE_ONLY_V1144='0',
                 LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139='0',
                 LMCACHE_PACKING_ABLATION_GPU_V1133='0',
                 LMCACHE_WRITE_CAP_EXPERIMENT_V1143='0')
    return flags
