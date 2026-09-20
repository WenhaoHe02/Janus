# SPDX-License-Identifier: Apache-2.0
"""Apply one run-script payload to all eight local TP workers."""

# Standard
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def run(rank: int, script: Path) -> object:
    """Execute one payload on one worker."""
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
    return ast.literal_eval(output)


def main() -> int:
    """Parse a script path, apply it to all ranks, and print results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=8) as pool:
        states = list(pool.map(lambda rank: run(rank, args.script), range(8)))
    print(f"worker_script={args.script.name} ranks=8/8 states={states}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
