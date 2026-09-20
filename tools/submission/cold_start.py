# SPDX-License-Identifier: Apache-2.0
"""Supervise build, private server startup, real-path warmup, and readiness."""

# Standard
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


def read_plan(path: Path) -> tuple[dict, str]:
    """Validate a trusted deployment plan and return its SHA256 fingerprint.

    The file contains explicit argv commands, not shell fragments. A model
    identity and loopback health URL are mandatory. Raises ValueError for an
    incomplete plan. Never reads an experiment trace during plan validation.
    """
    payload = path.read_bytes()
    plan = json.loads(payload)
    for key in ("model_revision", "gpu_arch", "tp_size", "precision", "shape_policy"):
        if not plan.get("identity", {}).get(key):
            raise ValueError(f"Missing identity.{key}")
    url = urlparse(plan["health_url"])
    if url.scheme != "http" or url.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Use a private loopback server during warmup")
    for key in ("build", "server", "warmup", "verify"):
        commands = plan.get(key)
        if key != "build":
            commands = [commands]
        if not isinstance(commands, list) or not commands:
            raise ValueError(f"Missing {key} command")
        for command in commands:
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(item, str) or not item for item in command)
            ):
                raise ValueError(f"{key} must contain argv arrays")
    for key in ("health_timeout_s", "command_timeout_s"):
        if not isinstance(plan.get(key), (int, float)) or plan[key] <= 0:
            raise ValueError(f"Positive {key} required")
    return plan, hashlib.sha256(payload).hexdigest()


def run_phase(
    command: list[str], env: dict, output: Path, name: str, timeout: float
) -> None:
    """Run an explicit argv, logging output; propagate failure or timeout."""
    with (output / f"{name}.log").open("xb") as log:
        subprocess.run(
            command,
            env=env,
            stdout=log,
            stderr=log,
            timeout=timeout,
            check=True,
            shell=False,
        )


def wait_health(server: subprocess.Popen, url: str, timeout: float) -> None:
    """Wait for internal HTTP health, failing if the server exits or times out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("Server exited before warmup")
        try:
            with urllib.request.urlopen(url, timeout=min(2, timeout)) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.2)
    raise TimeoutError("Server health timeout")


def validate_receipt(receipt: dict, artifact_id: str) -> None:
    """Require matched actual-content warmup with complete cohort coverage.

    This proves only the declared shapes/configuration were exercised; it does
    not prove all possible future shapes are compile-free.
    """
    if (
        receipt.get("artifact_id") != artifact_id
        or receipt.get("warmup_input_mode") != "actual_native_content"
        or receipt.get("status") != "complete"
        or type(receipt.get("covered_requests")) is not int
        or receipt["covered_requests"] <= 0
        or receipt.get("covered_requests") != receipt.get("expected_requests")
        or not receipt.get("manifest_sha256")
    ):
        raise ValueError("Missing or mismatched actual-path warmup proof")


def serve(plan_path: Path, output: Path) -> int:
    """Build and warm once before publishing readiness, then supervise server.

    Args:
        plan_path: Trusted deployment configuration; commands run as current user.
        output: New per-start directory, never reused or overwritten.
    Returns:
        Server exit code after ready, or raises on preparation failure.
    Notes:
        Route traffic only after READY.json exists. The private model's own
        /health is NOT deployment readiness. SIGTERM/interrupt removes READY.
        No host sysctl, disk mount, device binding or cache deletion is added.
    """
    plan, artifact_id = read_plan(plan_path)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ready = output / "READY.json"
    receipt_path = output / "warmup_receipt.json"
    inherited = dict(os.environ)
    if plan.get("isolate_lmcache_env", False):
        inherited = {
            key: value
            for key, value in inherited.items()
            if not key.startswith("LMCACHE_")
        }
    env = dict(inherited, **plan.get("env", {}))
    env.update(
        JANUS_ARTIFACT_ID=artifact_id,
        JANUS_WARMUP_RECEIPT=str(receipt_path),
        JANUS_START_DIRECTORY=str(output),
    )
    (output / "identity.json").write_text(
        json.dumps(dict(artifact_id=artifact_id, identity=plan["identity"]), indent=2)
    )
    server = None
    prior_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum: int, frame: object) -> None:
        """Unwind supervision so readiness is removed before terminating server."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    try:
        for index, command in enumerate(plan["build"]):
            run_phase(command, env, output, f"build_{index}", plan["command_timeout_s"])
        with (output / "server.log").open("xb") as log:
            server = subprocess.Popen(
                plan["server"],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=os.name == "posix",
            )
        wait_health(server, plan["health_url"], plan["health_timeout_s"])
        run_phase(plan["warmup"], env, output, "warmup", plan["command_timeout_s"])
        validate_receipt(json.loads(receipt_path.read_text()), artifact_id)
        run_phase(plan["verify"], env, output, "verify", plan["command_timeout_s"])
        if server.poll() is not None:
            raise RuntimeError("Server exited during preparation")
        temporary = output / "READY.tmp"
        temporary.write_text(
            json.dumps(
                dict(
                    artifact_id=artifact_id,
                    server_pid=server.pid,
                    warmup_receipt=str(receipt_path),
                    ready_at=time.time(),
                )
            )
        )
        temporary.replace(ready)
        print(json.dumps(dict(status="ready", file=str(ready))), flush=True)
        return server.wait()
    finally:
        ready.unlink(missing_ok=True)
        if server is not None:
            if os.name == "posix":
                # Only the process group created by this invocation is owned.
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(server.pid, signal.SIGKILL)
                else:
                    server.kill()
                server.wait()
        signal.signal(signal.SIGTERM, prior_handler)


def main() -> None:
    """Run the deployment gate; validation does not contact a server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        _, digest = read_plan(args.plan)
        print(json.dumps(dict(valid=True, artifact_id=digest)))
    else:
        if args.output is None:
            parser.error("--output is required")
        raise SystemExit(serve(args.plan, args.output))


if __name__ == "__main__":
    main()
