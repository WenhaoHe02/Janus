# SPDX-License-Identifier: Apache-2.0
"""Hot-load compact-width GPU owner gathering on all TP ranks."""

# Standard
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def run(rank: int) -> dict[str, object]:
    """Install the compact gather on one rank."""
    script = Path("/tmp/hotload_compact_gpu_owner_v432.py")
    output = subprocess.check_output(
        [
            "curl",
            "--max-time",
            "30",
            "-fsS",
            "-F",
            f"script=@{script}",
            f"http://127.0.0.1:{7000 + rank}/run_script",
        ],
        text=True,
    )
    value = ast.literal_eval(output)
    if not isinstance(value, dict):
        raise RuntimeError(f"rank {rank} returned a non-object result")
    return value


def main() -> int:
    """Install on all ranks and verify all acknowledgements."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(pool.map(run, range(8)))
    if not all(state.get("installed") is True for state in states):
        raise RuntimeError(f"compact GPU owner install failed: {states}")
    print(f"compact_gpu_owner ranks=8/8 states={states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
