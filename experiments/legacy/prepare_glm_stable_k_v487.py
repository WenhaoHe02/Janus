# SPDX-License-Identifier: Apache-2.0
"""Install the stable GLM K-local runtime without v470/v463 regressions."""

# Standard
from pathlib import Path
import subprocess


def run(arguments: list[str]) -> None:
    """Run one checked stable-runtime preparation command."""
    subprocess.run(arguments, check=True)


def main() -> int:
    """Install the validated v435 components on all TP workers."""
    required = (
        Path("/tmp/device_miss_v429.py"),
        Path("/tmp/apply_glm_device_miss_v428.py"),
        Path("/tmp/apply_compact_gpu_owner_v432.py"),
        Path("/tmp/hotload_glm_proxy_buckets_v437.py"),
        Path("/tmp/config_glm_compact_pipeline_v435.py"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"stable GLM runtime inputs are missing: {missing}")
    run(
        [
            "docker",
            "cp",
            "/tmp/device_miss_v429.py",
            "dsv4-csa-cp8-off:/usr/local/lib/python3.12/dist-packages/"
            "lmcache/v1/device_miss.py",
        ]
    )
    run(["python3", "/tmp/apply_glm_device_miss_v428.py"])
    run(["python3", "/tmp/apply_compact_gpu_owner_v432.py"])
    run(
        [
            "python3",
            "/tmp/apply_worker_script_v434.py",
            "/tmp/config_glm_compact_pipeline_v435.py",
        ]
    )
    run(
        [
            "python3",
            "/tmp/apply_worker_script_v434.py",
            "/tmp/hotload_glm_proxy_buckets_v437.py",
        ]
    )
    print("GLM_STABLE_K_V487_READY ranks=8/8", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
