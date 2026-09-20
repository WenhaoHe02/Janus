# SPDX-License-Identifier: Apache-2.0
"""Opt-in request/step NVTX scopes around public vLLM worker interfaces."""

# Standard
from functools import wraps
import os
from typing import Any


def install_model_step_nvtx(runner_class: type, nvtx: Any = None) -> bool:
    """Label execute/sample calls without synchronizing or retaining request history.

    Args:
        runner_class: GPUModelRunner exposing execute_model and sample_tokens.
        nvtx: Optional range_start/range_end provider, used by CPU contract tests.
    Returns:
        True if enabled and installed, False if the diagnostic flag is disabled.
    Raises:
        RuntimeError: These hooks were already installed.
        AttributeError: A required public runner interface is absent.
    Notes:
        Only one current label is retained per runner. An execute range measures
        host issue scope, not device completion; correlate GPU launches separately.
    """
    if os.getenv("LMCACHE_SERVING_DIAGNOSTIC_STEP_NVTX") != "1":
        return False
    if getattr(runner_class, "model_step_nvtx_installed", False):
        raise RuntimeError("model-step NVTX already installed")
    original_execute = runner_class.execute_model
    original_sample = runner_class.sample_tokens
    if nvtx is None:
        # Third Party
        import torch

        nvtx = torch.cuda.nvtx

    @wraps(original_execute)
    def execute(owner: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        scheduled = scheduler_output.num_scheduled_tokens
        identity = ",".join(str(key) for key in scheduled) or "none"
        label = f"{identity}/tokens={sum(scheduled.values())}"
        owner.model_step_nvtx_label = label
        handle = nvtx.range_start(f"GLM_STEP_V1019/{label}")
        try:
            return original_execute(owner, scheduler_output, *args, **kwargs)
        finally:
            nvtx.range_end(handle)

    @wraps(original_sample)
    def sample(owner: Any, *args: Any, **kwargs: Any) -> Any:
        label = getattr(owner, "model_step_nvtx_label", "none/tokens=0")
        handle = nvtx.range_start(f"GLM_SAMPLE_V1019/{label}")
        try:
            return original_sample(owner, *args, **kwargs)
        finally:
            nvtx.range_end(handle)

    runner_class.execute_model = execute
    runner_class.sample_tokens = sample
    runner_class.model_step_nvtx_installed = True
    return True
