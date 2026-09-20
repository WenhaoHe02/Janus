# SPDX-License-Identifier: Apache-2.0
"""Explicit prediction and packing ablation arms on a fixed Flash baseline."""

# Local
from flash_parallel_prediction_v1182 import environment as prediction_environment

ARMS = ('baseline', 'unsharded', 'no_read_cpu', 'no_read_gpu')


def environment(arm: str) -> dict[str, str]:
    """Return flags for one arm; reject unknown arms without changing defaults.

    Args:
        arm: baseline, unsharded, no_read_cpu, or no_read_gpu.
    Returns:
        Prediction/storage flags, changing only the requested mechanism.
    Raises:
        ValueError: The arm is not part of the declared comparison.
    """
    if arm not in ARMS:
        raise ValueError(f'Unsupported TTFT ablation arm: {arm}')
    flags = prediction_environment('unsharded' if arm == 'unsharded' else 'parallel')
    flags['LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139'] = '1' if arm.startswith('no_read_') else '0'
    flags['LMCACHE_PACKING_ABLATION_GPU_V1133'] = '1' if arm == 'no_read_gpu' else '0'
    return flags
