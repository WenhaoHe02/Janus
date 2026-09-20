# SPDX-License-Identifier: Apache-2.0
"""User-authorized, bounded 20→0→20 compaction-proactiveness diagnostic."""

# Standard
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
from typing import Any


def restore_setting(path: Path, original: str) -> dict:
    """Restore only our temporary value0; do not overwrite another admin's value.

    Return the before/after values. Raise RuntimeError if the current value is
    neither0 nor the saved original, or if readback differs after restoration.
    """
    current = path.read_text().strip()
    if current not in ("0", original):
        raise RuntimeError(
            f"External setting change detected: {current}; not overwritten"
        )
    if current == "0":
        path.write_text(original + "\n")
    restored = path.read_text().strip()
    if restored != original:
        raise RuntimeError(f"Restoration readback failed: {restored}")
    return dict(before=current, after=restored, wall_s=time.time())


def restoration_watchdog(
    connection: Any, path: str, original: str, log: str, timeout_s: int = 180
) -> None:
    """Restore on parent death/pipe closure or after180s, unless safely disarmed."""
    connection.send("armed")
    reason = f"deadline{timeout_s}s"
    try:
        if connection.poll(timeout_s):
            try:
                message = connection.recv()
            except EOFError:
                reason = "parent_pipe_closed"
            else:
                if message == "restored_and_verified":
                    Path(log).write_text(
                        json.dumps(dict(disarmed=True, wall_s=time.time()))
                    )
                    return
                reason = f"unexpected_parent_message:{message}"
        try:
            outcome = restore_setting(Path(path), original)
        except Exception as error:
            outcome = dict(error=str(error))
        Path(log).write_text(json.dumps(dict(reason=reason, outcome=outcome), indent=2))
    finally:
        connection.close()


def capture(root: Path, phase: str, repeat: int) -> dict:
    """Capture the existing unchanged109K request and preserve all raw output."""
    output = root / f"{phase}{repeat}"
    with (root / f"{phase}{repeat}.log").open("x") as log:
        subprocess.run(
            [
                "python3",
                "/tmp/capture_long_cpu_counters_v1025.py",
                "--host-cpu-census",
                "--output",
                str(output),
            ],
            stdout=log,
            stderr=log,
            check=True,
        )
    request = json.loads((output / "request.json").read_text())
    census = json.loads((output / "host_cpu_census.json").read_text())
    before, after = census["before"], census["after"]
    return dict(
        phase=phase,
        repeat=repeat,
        request_id=request["request_id"],
        ttft_ms=request["ttft_ms"],
        tpot_ms=request["tpot_ms"],
        counters_complete=request["cpu_sampling_complete"],
        proactiveness_before=before["compaction_proactiveness"],
        proactiveness_after=after["compaction_proactiveness"],
        compaction_delta={
            k: after["vmstat"][k] - v
            for k, v in before["vmstat"].items()
            if k.startswith("compact_")
        },
        kcompactd_cpu_s=sum(
            t["cpu_s"] for t in census["deltas"] if t["name"].startswith("kcompactd")
        ),
    )


def main() -> None:
    """Measure three requests per phase, restore20 before final measurements.

    The only host mutation is temporarily setting vm.compaction_proactiveness0,
    explicitly approved by the user. No cache drop, memory compaction trigger,
    disk changes, model restart or algorithm modification is performed. Writing
    the restored nonzero value may itself trigger proactive compaction; the
    after phase is a reversal diagnostic, not steady-state serving data.
    """
    # Standard
    import fcntl

    # Local
    from diagnose_serving_decode_v973 import assert_idle
    from run_original_serving_v957 import verify

    lock = Path("/tmp/remaining_models_v957.launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert_idle()
    proof = verify("k_local", "serving_glm_configuration_v766", allow_diagnostic=True)
    if (
        proof["container_id"]
        != "f1011ede1a5fd1934bcfc75928d5f56316a9c58a03105be94ef7aa96ce79348f"
    ):
        raise RuntimeError("Expected unchanged GLM deployment")
    path = Path("/proc/sys/vm/compaction_proactiveness")
    original = path.read_text().strip()
    if original != "20":
        raise RuntimeError(f"Expected preauthorized original value20, got{original}")
    root = Path("/tmp/glm_compaction_ab_v1030")
    root.mkdir(exist_ok=False)
    (root / "source.py").write_bytes(Path(__file__).read_bytes())
    (root / "deployment.json").write_text(json.dumps(proof, indent=2))
    (root / "setting_before.json").write_text(
        json.dumps(dict(path=str(path), original=original))
    )
    rows = []

    def record(phase: str, repeat: int) -> None:
        """Persist each completed request before proceeding to the next phase."""
        row = capture(root, phase, repeat)
        rows.append(row)
        (root / "RESULTS.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)

    for repeat in range(3):
        record("original20", repeat)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    watchdog = context.Process(
        target=restoration_watchdog,
        args=(child, str(path), original, str(root / "watchdog.json")),
    )
    watchdog.start()
    child.close()
    if not parent.poll(10) or parent.recv() != "armed":
        parent.close()
        watchdog.join(timeout=3)
        raise RuntimeError("Watchdog did not arm; setting not changed")
    try:
        if path.read_text().strip() != original:
            raise RuntimeError("Setting changed before experiment; not modified")
        path.write_text("0\n")
        if path.read_text().strip() != "0":
            raise RuntimeError("Temporary setting readback failed")
        (root / "disabled_at.json").write_text(
            json.dumps(dict(wall_s=time.time(), value="0"))
        )
        for repeat in range(3):
            if path.read_text().strip() != "0":
                raise RuntimeError("Temporary setting unexpectedly changed")
            record("disabled0", repeat)
    finally:
        try:
            restored = restore_setting(path, original)
            (root / "restored.json").write_text(json.dumps(restored, indent=2))
            parent.send("restored_and_verified")
        finally:
            parent.close()
            watchdog.join(timeout=5)
    for repeat in range(3):
        if path.read_text().strip() != original:
            raise RuntimeError("Original value not retained after restoration")
        record("restored20", repeat)
    (root / "COMPLETE.json").write_text(
        json.dumps(
            dict(original=original, final=path.read_text().strip(), requests=rows),
            indent=2,
        )
    )
    print("COMPACTION_AB_COMPLETE_ORIGINAL20_RESTORED", flush=True)


if __name__ == "__main__":
    main()
