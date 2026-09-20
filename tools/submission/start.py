# SPDX-License-Identifier: Apache-2.0
"""Single versionless startup entry for the three Janus model families."""

# Standard
import argparse
import json
from pathlib import Path

# Local
from cold_start import read_plan, serve


def validate_model_plan(path: Path, model: str) -> dict:
    """Require explicit model identity, no query sharding and native build evidence.

    Args:
        path: Trusted deployment plan for cold_start.py.
        model: flash, pro or glm.
    Returns:
        Validated plan; raises ValueError for conflicting/missing configuration.
    Does not launch a server or change host configuration.
    """
    plan, _ = read_plan(path)
    if plan["identity"].get("model_family") != model:
        raise ValueError("Plan model_family differs from --model")
    if plan["identity"]["tp_size"] != 8:
        raise ValueError("This artifact was designed for TP8")
    env = plan.get("env", {})
    if env.get("PYTHONHASHSEED") != "0":
        raise ValueError("Use PYTHONHASHSEED=0 consistently before worker creation")
    for key in ("LMCACHE_PROXY_INDEXER_CP", "LMCACHE_TRUE_INDEXER_CP"):
        if env.get(key) != "0":
            raise ValueError(f"Set {key}=0 explicitly")
    for key in ("JANUS_COUNTED_BUILD_RECEIPT", "JANUS_COUNTED_SOURCE_ROOT"):
        if not env.get(key):
            raise ValueError(f"Missing native startup setting: {key}")
    if not any(
        "build_native.py" in arg for command in plan["build"] for arg in command
    ):
        raise ValueError("Build counted-read native code before serving")
    return plan


def main() -> None:
    """Validate or run one pinned startup plan; no historical version selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("flash", "pro", "glm"), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    validate_model_plan(args.plan, args.model)
    if args.check_only:
        print(json.dumps(dict(model=args.model, valid=True)))
        return
    if args.output is None:
        parser.error("--output is required for startup")
    raise SystemExit(serve(args.plan, args.output))


if __name__ == "__main__":
    main()
