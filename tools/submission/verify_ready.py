# SPDX-License-Identifier: Apache-2.0
"""Check all workers and drained writes after startup warmup, before readiness."""

# Standard
import argparse
import json
import os
from pathlib import Path

# Local
from startup_io import StartupClient


def main() -> None:
    """Check exact profile and loaded native digest, without generating tokens."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--rank-endpoints", nargs=8, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--worker-env", type=Path, required=True)
    args = parser.parse_args()
    client = StartupClient(
        args.endpoint,
        args.rank_endpoints,
        args.model_name,
        Path(os.environ["JANUS_START_DIRECTORY"]) / "server.log",
    )
    expected = json.loads(args.worker_env.read_text())
    expected["JANUS_ARTIFACT_ID"] = os.environ["JANUS_ARTIFACT_ID"]
    client.verify_workers(expected)
    client.verify_loaded_native(Path(os.environ["JANUS_COUNTED_BUILD_RECEIPT"]))
    client.drain()
    print(json.dumps(dict(ranks=8, native_loaded=True, writers_drained=True)))


if __name__ == "__main__":
    main()
