# SPDX-License-Identifier: Apache-2.0
"""Install step labels in a fresh vLLM module before model workers import it."""

# Standard
import argparse
from pathlib import Path
import shutil


def main() -> None:
    """Back up and patch only the diagnostic runner entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    target = args.vllm_root / "v1/worker/gpu_model_runner.py"
    source = target.read_text()
    backup = target.with_suffix(".py.before_step_nvtx_v1019")
    if backup.exists() or "class GPUModelRunner(" not in source:
        raise RuntimeError("Require fresh supported GPUModelRunner")
    helper = args.helper.read_text()
    patched = source + '''
# MODEL_STEP_NVTX_V1019
import os as step_nvtx_environment_v1019
if step_nvtx_environment_v1019.getenv("LMCACHE_SERVING_DIAGNOSTIC_STEP_NVTX") == "1":
    from lmcache.integration.vllm.model_step_nvtx import install_model_step_nvtx
    install_model_step_nvtx(GPUModelRunner)
'''
    compile(helper, str(args.helper), "exec")
    compile(patched, str(target), "exec")
    shutil.copy2(target, backup)
    (args.lmcache_root / "integration/vllm/model_step_nvtx.py").write_text(helper)
    target.write_text(patched)
    print("MODEL_STEP_NVTX_V1019_INSTALLED", flush=True)


if __name__ == "__main__":
    main()
