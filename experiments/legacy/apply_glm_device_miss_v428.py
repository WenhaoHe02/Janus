# SPDX-License-Identifier: Apache-2.0
"""Install the GLM device-counted correction experiment on all TP ranks."""

# Standard
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def run_script(rank: int, script: Path) -> dict[str, object]:
    """Run one admin script on a rank and return its result."""
    output = subprocess.check_output(
        [
            "curl",
            "--max-time",
            "90",
            "-fsS",
            "-F",
            f"script=@{script}",
            f"http://127.0.0.1:{7000 + rank}/run_script",
        ],
        text=True,
    )
    result = ast.literal_eval(output)
    if not isinstance(result, dict):
        raise RuntimeError(f"rank {rank} returned a non-object result")
    return result


def main() -> int:
    """Install counted I/O first, then the GLM manager integration."""
    checks = (
        (Path("/tmp/install_counted_ops_only_v208.py"), "installed"),
        (Path("/tmp/hotload_glm_device_miss_v425.py"), "installed"),
    )
    for script, expected in checks:
        with ThreadPoolExecutor(max_workers=8) as pool:
            states = list(pool.map(lambda rank: run_script(rank, script), range(8)))
        if not all(state.get(expected) is True for state in states):
            raise RuntimeError(f"{script.name} failed: {states}")
        print(f"{script.name}: ranks=8/8 states={states}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
