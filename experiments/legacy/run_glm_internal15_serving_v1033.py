# SPDX-License-Identifier: Apache-2.0
"""Run frozen native internal15 serving after the three-point SSD comparison."""

# Standard
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import time

# Local
from compaction_ab_v1030 import restoration_watchdog, restore_setting
from diagnose_serving_decode_v973 import assert_idle
from run_original_serving_v957 import formal_rows, verify


def audit_boundaries(output: Path, rows: list[dict]) -> list[dict]:
    """Join completed request IDs to actual engine hits without changing metrics.

    Args:
        output: Completed serving-rate directory containing raw requests/logs.
        rows: Frozen native inputs defining each independent cache snapshot.
    Returns:
        Actual prompt/engine/hit/load counters and pass/fail per request.
    Raises:
        ValueError: A completed record refers to an unknown source identity.
    """
    expected = {row["source_request_id"]: row for row in rows}
    records = [
        json.loads(line)
        for line in (output / "requests.jsonl").read_text().splitlines()
    ]
    lines = (output / "server_after.log").read_text().splitlines()
    result = []
    for record in records:
        row = expected[record["source_request_id"]]
        request_id = record.get("request_id")
        found = []
        for line in lines:
            if request_id and request_id in line and "LMCache hit tokens:" in line:
                match = re.search(
                    r"Total tokens (\d+), Inference Engine computed tokens: (\d+), "
                    r"LMCache hit tokens: (\d+), need to load: (\d+)",
                    line,
                )
                if match:
                    found.append(tuple(int(value) for value in match.groups()))
        history = row["reusable_input_prefix_tokens"] // 256 * 256
        counters = (row["prompt_tokens"], 0, history, history)
        result.append(
            dict(
                source_request_id=row["source_request_id"],
                request_id=request_id,
                expected=counters,
                observed=found,
                exact_boundary_verified=(
                    record.get("status") == "success" and found == [counters]
                ),
            )
        )
    return result


def main() -> None:
    """Run selected K/NP/SSD/DRAM arms at .015/.01/.005, restoring the host setting.

    Original15 IDs and O remain fixed. No decode correction, no query CP, no
    profiling. Independent cache tags prevent future seeds preloading earlier
    appends. Shape warmup is reused only within the same container/cohort.
    """
    # Standard
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("k_local", "no_prefetch", "ssd_only", "dram"),
        default=["k_local", "no_prefetch", "ssd_only", "dram"],
    )
    parser.add_argument("--wait-for-complete", type=Path)
    parser.add_argument("--predecessor-pid", type=int)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Duplicate comparison arms")
    if args.wait_for_complete:
        print(
            f"WAIT_PREDECESSOR complete={args.wait_for_complete} pid={args.predecessor_pid}",
            flush=True,
        )
        while not args.wait_for_complete.exists():
            if args.predecessor_pid is not None:
                process = Path(f"/proc/{args.predecessor_pid}/cmdline")
                if (
                    not process.exists()
                    or b"run_glm_internal15_serving_v1033.py"
                    not in process.read_bytes()
                ):
                    raise RuntimeError(
                        "Predecessor stopped without successful completion; no new arm started"
                    )
            time.sleep(20)
        preceding = json.loads(args.wait_for_complete.read_text())
        expected = {
            (arm, rate)
            for arm in ("k_local", "no_prefetch", "ssd_only")
            for rate in (0.015, 0.01, 0.005)
        }
        if {(item["arm"], item["rate"]) for item in preceding} != expected or any(
            item["successes"] != 15 or item["failures"] for item in preceding
        ):
            raise RuntimeError("Predecessor did not finish all three arms and rates")
        if args.predecessor_pid is not None:
            process = Path(f"/proc/{args.predecessor_pid}/cmdline")
            while process.exists() and b"run_glm_internal15_serving_v1033.py" in process.read_bytes():
                time.sleep(0.2)
    lock = Path("/tmp/remaining_models_v957.launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = Path("/tmp/glm_long_threeway_v1031_ssd32_measure")
    complete = json.loads((previous / "COMPLETE.json").read_text())
    if (
        len(complete["requests"]) != 9
        or complete["final_compaction_proactiveness"] != "20"
    ):
        raise RuntimeError("SSD comparison/restoration incomplete")
    assert_idle()
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    rows = formal_rows(manifest)
    if (
        len(rows) != 15
        or not manifest["historical_reasoning_retained"]
        or any(row["reusable_input_prefix_tokens"] // 256 * 256 < 65536 for row in rows)
    ):
        raise ValueError("Exactly15 fixed native64K+ full-history inputs required")
    setting = Path("/proc/sys/vm/compaction_proactiveness")
    original = setting.read_text().strip()
    if original != "20":
        raise RuntimeError("Previous scoped host setting not restored")
    args.output.mkdir(exist_ok=False)
    (args.output / "manifest.json").write_bytes(
        (args.inputs / "manifest.json").read_bytes()
    )
    (args.output / "coordinator.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "run_scope.json").write_text(
        json.dumps(
            dict(
                arms=args.arms,
                rates=[0.015, 0.01, 0.005],
                predecessor_complete=str(args.wait_for_complete)
                if args.wait_for_complete
                else None,
                dram_cpu_gib_per_rank=128 if "dram" in args.arms else None,
            ),
            indent=2,
        )
    )
    (args.output / "setting_before.json").write_text(
        json.dumps(dict(original=original, temporary="0"))
    )
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    watchdog = context.Process(
        target=restoration_watchdog,
        args=(
            child,
            str(setting),
            original,
            str(args.output / "watchdog.json"),
            28800,
        ),
    )
    watchdog.start()
    child.close()
    if not parent.poll(10) or parent.recv() != "armed":
        parent.close()
        raise RuntimeError("Host restoration watchdog did not arm")
    summaries = []
    started = time.monotonic()
    try:
        setting.write_text("0\n")
        for arm in args.arms:
            assert_idle()
            # A stopped experiment's inherited helper can otherwise retain the
            # old launch lock. Stop only this idle, fully archived container.
            subprocess.run(
                ["docker", "stop", "-t", "120", "dsv4-csa-cp8-off"], check=True
            )
            with Path("/tmp/glm_cluster_relaunch_v306.lock").open() as launch_lock:
                deadline = time.monotonic() + 180
                while True:
                    try:
                        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Prior container helper retained launch lock"
                            )
                        time.sleep(2)
                    else:
                        fcntl.flock(launch_lock, fcntl.LOCK_UN)
                        break
            startup = (
                "/tmp/startup_glm_serving_deviceowner_v1006.sh"
                if arm == "k_local"
                else "/tmp/startup_glm_serving_ssd32_v1032.sh"
                if arm == "ssd_only"
                else "/tmp/startup_glm_serving_baseline_v1031.sh"
            )
            environment = dict(
                os.environ,
                LMCACHE_EXPERIMENT_FORCE_RELAUNCH="1",
                LMCACHE_EXPERIMENT_STARTUP_SCRIPT=startup,
                LMCACHE_ABLATION_MAX_MODEL_LEN="524288",
                LMCACHE_ABLATION_KV_CACHE_MEMORY_BYTES="21474836480",
                LMCACHE_CPU_PER_RANK_GB="128",
                LMCACHE_CPU_FIRST_RANK_GB="128",
            )
            with (args.output / f"{arm}_relaunch.log").open("x") as log:
                subprocess.run(
                    ["bash", "/tmp/relaunch_glm_serving_v957.sh", arm, "ttft"],
                    env=environment,
                    stdout=log,
                    stderr=log,
                    check=True,
                )
            proof = verify(
                arm,
                "serving_glm_configuration_v766",
                allow_device_owner=(arm == "k_local"),
            )
            if arm == "dram":
                config = subprocess.check_output(
                    [
                        "docker",
                        "exec",
                        "dsv4-csa-cp8-off",
                        "cat",
                        "/tmp/lmcache_ssd_tutti_kvobj.yaml",
                    ],
                    text=True,
                )
                if not re.search(
                    r"^local_cpu:\s*true\s*$", config, re.M
                ) or not re.search(
                    r"^max_local_cpu_size:\s*128(?:\.0)?\s*$", config, re.M
                ):
                    raise RuntimeError("Expected DRAM cache tier with128GiB/rank")
                (args.output / "dram_backend_config.yaml").write_text(config)
            (args.output / f"{arm}_deployment.json").write_text(
                json.dumps(proof, indent=2)
            )
            prepared = None
            for rate in (0.015, 0.01, 0.005):
                if (
                    setting.read_text().strip() != "0"
                    or time.monotonic() - started > 27000
                ):
                    raise RuntimeError("Matched host-policy scope ended")
                output = args.output / f"{arm}_{str(rate).replace('.', 'p')}"
                command = [
                    "python3",
                    "/tmp/run_original_serving_v957.py",
                    "--inputs",
                    str(args.inputs),
                    "--output",
                    str(output),
                    "--arm",
                    arm,
                    "--rate",
                    str(rate),
                    "--configuration-module",
                    "serving_glm_configuration_v766",
                    "--isolate-request-cache",
                ]
                if arm == "k_local":
                    command.append("--device-owner")
                if prepared is not None:
                    command.extend(["--reuse-shapes", str(prepared)])
                print(f"RUN_SERVING arm={arm} rate={rate} requests=15", flush=True)
                with (args.output / f"{output.name}.log").open("x") as log:
                    subprocess.run(command, stdout=log, stderr=log, check=True)
                with (output / "server_after.log").open("x") as log:
                    subprocess.run(
                        ["docker", "logs", "dsv4-csa-cp8-off"],
                        stdout=log,
                        stderr=log,
                        check=True,
                    )
                with (output / "container_after.json").open("x") as file:
                    subprocess.run(
                        ["docker", "inspect", "dsv4-csa-cp8-off"],
                        stdout=file,
                        check=True,
                    )
                boundaries = audit_boundaries(output, rows)
                (output / "boundary_audit.json").write_text(
                    json.dumps(boundaries, indent=2)
                )
                result = json.loads((output / "COMPLETE.json").read_text())
                if (
                    result["successes"] != 15
                    or result["failures"]
                    or not all(item["exact_boundary_verified"] for item in boundaries)
                ):
                    raise RuntimeError(
                        "Serving failed or independent cache boundary changed; raw records retained"
                    )
                if setting.read_text().strip() != "0":
                    raise RuntimeError(
                        "Host policy changed during serving; do not pool this batch"
                    )
                summaries.append(dict(arm=arm, rate=rate, output=str(output), **result))
                (args.output / "RESULTS.json").write_text(
                    json.dumps(summaries, indent=2)
                )
                prepared = output / "prepared.json"
                print(f"COMPLETE_SERVING arm={arm} rate={rate} requests=15", flush=True)
    finally:
        try:
            restored = restore_setting(setting, original)
            (args.output / "restored.json").write_text(json.dumps(restored, indent=2))
            parent.send("restored_and_verified")
        finally:
            parent.close()
            watchdog.join(timeout=5)
    (args.output / "COMPLETE.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
