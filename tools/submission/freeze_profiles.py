# SPDX-License-Identifier: Apache-2.0
"""Freeze serving flag dictionaries as configuration, not old startup programs."""

# Standard
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def freeze(legacy: Path, output: Path) -> dict:
    """Evaluate the two source-only configuration functions into a JSON table.

    Args:
        legacy: Audited configuration helper directory from source assembly.
        output: New JSON configuration path, never overwritten.
    Returns:
        Per-family and per-arm flags without decode-read bypasses.
    Run in a fresh process; imports only the reviewed pure configuration modules.
    """
    sys.path.insert(0, str(legacy.resolve()))
    result = {}
    origins = {}
    for family, module in (
        ("dsv4", "serving_ds_configuration_v890"),
        ("glm", "serving_glm_configuration_v766"),
    ):
        config = importlib.import_module(module)
        result[family] = {}
        for arm in ("k_local", "no_prefetch", "ssd_only", "dram"):
            flags = config.environment(arm)
            if any("ASSUMED_HBM" in key for key in flags):
                raise ValueError("Configuration bypasses decode reads")
            result[family][arm] = flags
        origins[family] = hashlib.sha256(
            (legacy / (module + ".py")).read_bytes()
        ).hexdigest()
    result["source_sha256"] = origins
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.legacy, args.output)
