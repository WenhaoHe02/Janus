# SPDX-License-Identifier: Apache-2.0
"""Run selected DS serving arms after an explicitly verified predecessor."""

# Standard
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def snapshot_previous_server(output: Path) -> None:
    """Capture prior task containers, including an incompletely launched K arm.

    Args:
        output: New log path, never overwrite an existing experiment artifact.
    Raises:
        CalledProcessError: An existing container's logs cannot be captured.
    """
    with output.open("x") as log:
        for name in ("dsv4-csa-cp8-off", "dsv4-csa-cp8-on"):
            present = subprocess.run(
                ["docker", "inspect", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if present.returncode == 0:
                log.write(f"Previous task container: {name}\n")
                log.flush()
                subprocess.run(
                    ["docker", "logs", name], stdout=log, stderr=log, check=True
                )


def snapshot_native_inputs(inputs: Path, output: Path) -> dict:
    """Archive the frozen manifest and reject changed tokens across comparison arms.

    Args:
        inputs: Native model input directory with manifest and int32 token file.
        output: Experiment directory in which the first manifest is preserved.
    Returns:
        SHA256 identities of the actual native manifest and token file.
    Raises:
        ValueError: Token bytes disagree with their manifest or an earlier arm.
    """
    manifest_bytes = (inputs / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256()
    with (inputs / "tokens.i32le").open("rb") as token_file:
        for block in iter(lambda: token_file.read(1024 * 1024), b""):
            digest.update(block)
    token_hash = digest.hexdigest()
    if token_hash != manifest["token_file_sha256"]:
        raise ValueError("Native token file differs from frozen manifest")
    target = output / "manifest.json"
    if target.exists():
        if target.read_bytes() != manifest_bytes:
            raise ValueError("Native manifest changed between comparison arms")
    else:
        target.write_bytes(manifest_bytes)
    return dict(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        token_file_sha256=token_hash,
    )


def check_predecessor(root: Path, arms: list[str]) -> None:
    """Require completed, identity-matched fifteen-request predecessor batches.

    Args:
        root: Completed prior model's experiment, never mixed old-version roots.
        arms: All paths that must finish before switching models.
    Raises:
        ValueError: Missing completion, failed request, or mismatched identities.
    """
    if not (root / "COMPLETE.json").is_file():
        raise ValueError("Predecessor coordinator has not completed")
    source_index = root / "COHORT_SOURCES.json"
    references = {}
    if source_index.exists():
        entries = json.loads(source_index.read_text())
        references = {
            (entry["arm"], entry["requested_slot"]): Path(entry["batch"])
            for entry in entries
        }
        if len(references) != len(entries):
            raise ValueError("Duplicate referenced batch in retained-result index")
    expected_ids = None
    for arm in arms:
        for rate in ("0p015", "0p01", "0p005"):
            folder = references[(arm, rate)] if references else root / f"{arm}_{rate}"
            result = json.loads((folder / "COMPLETE.json").read_text())
            audit = json.loads((folder / "boundary_audit.json").read_text())
            ids = [item["source_request_id"] for item in audit]
            if (
                result["successes"] != 15
                or result["failures"]
                or len(ids) != 15
                or len(set(ids)) != 15
                or not all(item["exact_boundary_verified"] for item in audit)
            ):
                raise ValueError(f"Incomplete predecessor: {folder}")
            if expected_ids is not None and ids != expected_ids:
                raise ValueError("Predecessor arms differ in request identity/order")
            expected_ids = ids


def activate_launch_files(bundle: Path, output: Path) -> None:
    """Back up and activate only three explicitly prepared DS startup files.

    Args:
        bundle: Read-only preparation audit containing original/staged file hashes.
        output: New experiment directory for recoverable startup backups.
    Raises:
        ValueError: Files changed since preparation; never overwrite unknown edits.
    """
    import hashlib
    import shutil

    audit = json.loads(bundle.read_text())
    for row in audit["launch_files"]:
        actual = Path(row["target"])
        staged = Path(row["staged"])
        actual_hash = hashlib.sha256(actual.read_bytes()).hexdigest()
        staged_hash = hashlib.sha256(staged.read_bytes()).hexdigest()
        if staged_hash != row["staged_sha256"]:
            raise ValueError(f"Staged file changed: {staged}")
        if actual_hash not in (row["original_sha256"], row["staged_sha256"]):
            raise ValueError(f"Preserve newer startup edits: {actual}")
    backup = output / "launch_backups"
    backup.mkdir()
    for row in audit["launch_files"]:
        actual = Path(row["target"])
        shutil.copy2(actual, backup / actual.name)
        ownership = actual.stat()
        descriptor, name = tempfile.mkstemp(
            prefix=actual.name + ".install-", dir=actual.parent
        )
        os.close(descriptor)
        temporary = Path(name)
        try:
            shutil.copy2(row["staged"], temporary)
            if hasattr(os, "chown"):
                os.chown(temporary, ownership.st_uid, ownership.st_gid)
            temporary.replace(actual)
        finally:
            temporary.unlink(missing_ok=True)


def validate_backend(arm: str, config: dict) -> None:
    """Validate the requested buffered SSD or128GiB DRAM backend contract.

    Args:
        arm: Serving arm identifier.
        config: Actual loaded LMCache configuration.
    Raises:
        ValueError: SSD/DRAM backend, allocation, or direct-I/O policy differs.
    """
    if arm == "ssd_only" and not (
        config.get("local_cpu") is False
        and config.get("retrieve_locations") == ["LocalDiskBackend"]
        and config.get("store_location") == "LocalDiskBackend"
        and config.get("extra_config", {}).get("use_odirect") is False
        and float(config.get("max_local_cpu_size", 0)) == 32
    ):
        raise ValueError("SSD must be buffered LocalDiskBackend with32GiB staging")
    if arm == "dram" and not (
        config.get("local_cpu") is True
        and float(config.get("max_local_cpu_size", 0)) == 128
        and config.get("retrieve_locations") == ["LocalCPUBackend"]
        and config.get("store_location") == "LocalCPUBackend"
    ):
        raise ValueError("DRAM must be real128GiB LocalCPUBackend per rank")


def main() -> None:
    """Launch DS once per arm, prepare native inputs, then measure selected slots.

    Run on GPU002 after the declared predecessor or authorized cancellation.
    Existing output is never overwritten. Host compaction must match the explicit
    argument; its restoration belongs to the calling scope. Does not modify
    original IDs/O or algorithms. Startup patches must be deployed beforehand.
    """
    # Standard
    import fcntl

    # Local
    from diagnose_serving_decode_v973 import assert_idle
    from run_glm_internal15_serving_v1033 import audit_boundaries
    from run_original_serving_v957 import formal_rows, verify

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("flash", "pro"), required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("k_local", "no_prefetch", "dram", "ssd_only"),
        default=["k_local", "no_prefetch", "dram"],
    )
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--cancelled-predecessor-proof", type=Path)
    parser.add_argument("--host-compaction", choices=("0", "20"), default="20")
    parser.add_argument("--reuse-first-container", default="")
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        choices=(0.005, 0.01, 0.015),
        default=[0.015, 0.01, 0.005],
    )
    parser.add_argument(
        "--additional-predecessor", type=Path, action="append", default=[]
    )
    parser.add_argument("--predecessor-arms", nargs="+", default=[])
    parser.add_argument(
        "--bundle", type=Path, default=Path("/tmp/ds_retest_v1042/preparation.json")
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--public-reconstruction", action="store_true")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Duplicate serving arms")
    if len(set(args.rates)) != len(args.rates):
        raise ValueError("Duplicate arrival slots")
    if args.cancelled_predecessor_proof:
        cancellation = json.loads(args.cancelled_predecessor_proof.read_text())
        if (
            args.predecessor
            or cancellation.get("user_priority") != "flash_first_20260914"
        ):
            raise ValueError("Invalid explicit user-priority handoff")
        if any(Path(f"/proc/{pid}").exists() for pid in cancellation["cancelled_pids"]):
            raise RuntimeError("Cancelled GLM process still exists; do not switch")
    elif args.predecessor and args.predecessor_arms:
        check_predecessor(args.predecessor, args.predecessor_arms)
    else:
        raise ValueError(
            "Completed predecessor or explicit cancellation proof required"
        )
    for predecessor in args.additional_predecessor:
        check_predecessor(predecessor, args.predecessor_arms)
    setting = Path("/proc/sys/vm/compaction_proactiveness")
    if setting.read_text().strip() != args.host_compaction:
        raise ValueError("Actual host compaction differs from experiment scope")
    lock = Path("/tmp/remaining_models_v957.launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert_idle()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "coordinator.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "selection.json").write_bytes(args.selection.read_bytes())
    (args.output / "setting_before.json").write_text(
        json.dumps(dict(original=args.host_compaction, temporary=args.host_compaction))
    )
    (args.output / "run_scope.json").write_text(
        json.dumps(
            dict(
                model_kind=args.model_kind,
                arms=args.arms,
                rates=args.rates,
                public_reconstruction=args.public_reconstruction,
                checkpoint=str(args.checkpoint),
                predecessor=str(args.predecessor),
                additional_predecessors=[str(p) for p in args.additional_predecessor],
            ),
            indent=2,
        )
    )
    activate_launch_files(args.bundle, args.output)
    summaries = []
    # User deferred DS SSD-only for discussion; do not start it automatically.
    for arm_index, arm in enumerate(args.arms):
        assert_idle()
        snapshot_previous_server(args.output / f"{arm}_previous_server.log")
        environment = dict(
            os.environ,
            LMCACHE_SSD_LOCAL_CPU_GB="32",
            LMCACHE_CPU_PER_RANK_GB="128",
            LMCACHE_CPU_FIRST_RANK_GB="128",
            LMCACHE_EXPERIMENT_STARTUP_SCRIPT=os.environ.get(
                "LMCACHE_EXPERIMENT_STARTUP_SCRIPT", "/tmp/startup_ds_serving_v916.sh"
            ),
            LMCACHE_ABLATION_MAX_MODEL_LEN="524288",
        )
        with (args.output / f"{arm}_relaunch.log").open("x") as log:
            if arm_index == 0 and args.reuse_first_container:
                info = json.loads(
                    subprocess.check_output(["docker", "inspect", "dsv4-csa-cp8-off"])
                )[0]
                if (
                    info["Id"] != args.reuse_first_container
                    or not info["State"]["Running"]
                ):
                    raise ValueError("Ready container identity changed; cannot reuse")
                log.write(
                    json.dumps(dict(reused_ready_container=info["Id"], no_restart=True))
                )
            else:
                subprocess.run(
                    [
                        "bash",
                        "/tmp/relaunch_ds_serving_v892.sh",
                        args.model_kind,
                        "k_owner" if arm == "k_local" else arm,
                    ],
                    env=environment,
                    stdout=log,
                    stderr=log,
                    check=True,
                )
        proof = verify(arm, "serving_ds_configuration_v890")
        (args.output / f"{arm}_deployment.json").write_text(json.dumps(proof, indent=2))
        config = json.loads(
            subprocess.check_output(
                [
                    "docker",
                    "exec",
                    "dsv4-csa-cp8-off",
                    "python3",
                    "-c",
                    "import json,yaml;"
                    "print(json.dumps(yaml.safe_load("
                    "open('/tmp/lmcache_ssd_tutti_kvobj.yaml'))))",
                ],
                text=True,
            )
        )
        validate_backend(arm, config)
        (args.output / f"{arm}_backend.json").write_text(json.dumps(config, indent=2))
        if not args.inputs.exists():
            with (args.output / "native_inputs.log").open("x") as log:
                subprocess.run(
                    [
                        "python3",
                        "/tmp/build_ds_internal15_v1042.py",
                        "--model-kind",
                        args.model_kind,
                        "--selection",
                        str(args.selection),
                        "--checkpoint",
                        str(args.checkpoint),
                        "--output",
                        str(args.inputs),
                    ]
                    + (
                        ["--public-reconstruction"]
                        if args.public_reconstruction
                        else []
                    ),
                    stdout=log,
                    stderr=log,
                    check=True,
                )
        manifest = json.loads((args.inputs / "manifest.json").read_text())
        input_identity = snapshot_native_inputs(args.inputs, args.output)
        (args.output / f"{arm}_native_identity.json").write_text(
            json.dumps(input_identity, indent=2)
        )
        rows = formal_rows(manifest)
        chosen = json.loads(args.selection.read_text())["requests"]
        if args.public_reconstruction:
            from build_public_glm_native_v1046 import safe_source_id

            selection = json.loads(args.selection.read_text())
            if (
                selection.get("replay_kind") != "authorized_trajectory_reconstruction"
                or manifest.get("replay_kind") != selection["replay_kind"]
                or manifest.get("dataset") != selection["dataset"]
            ):
                raise ValueError("Public replay provenance or dataset differs")
            expected_ids = [
                safe_source_id(selection["dataset"], r["source_id"]) for r in chosen
            ]
            expected_aliases = dict(
                zip(expected_ids, [r["source_id"] for r in chosen], strict=True)
            )
            if manifest.get("source_id_alias_map") != expected_aliases:
                raise ValueError("Public source alias mapping differs")
        else:
            expected_ids = [r["source_id"] for r in chosen]
        if (
            manifest.get("model_kind") != args.model_kind
            or len(rows) != 15
            or [r["source_request_id"] for r in rows] != expected_ids
            or [r["output_budget"] for r in rows] != [r["O"] for r in chosen]
        ):
            raise ValueError("DS native manifest changed original IDs/O")
        logs = subprocess.check_output(
            ["docker", "logs", "dsv4-csa-cp8-off"], stderr=subprocess.STDOUT, text=True
        )
        capacity = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", logs)
        needed = max(r["prompt_tokens"] + r["output_budget"] for r in rows)
        if not capacity or min(int(n.replace(",", "")) for n in capacity) < needed:
            raise ValueError("DS GPU capacity is insufficient before seed")
        host_value = args.host_compaction
        prepared = None
        for rate in args.rates:
            if setting.read_text().strip() != host_value:
                raise RuntimeError("Host policy changed")
            folder = args.output / f"{arm}_{str(rate).replace('.', 'p')}"
            command = [
                "python3",
                "/tmp/run_original_serving_v957.py",
                "--inputs",
                str(args.inputs),
                "--output",
                str(folder),
                "--arm",
                arm,
                "--rate",
                str(rate),
                "--configuration-module",
                "serving_ds_configuration_v890",
                "--isolate-request-cache",
            ]
            if prepared:
                command.extend(["--reuse-shapes", str(prepared)])
            print(f"RUN_SERVING arm={arm} rate={rate} requests=15", flush=True)
            with (args.output / f"{folder.name}.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=log, check=True)
            with (folder / "server_after.log").open("x") as log:
                subprocess.run(
                    ["docker", "logs", "dsv4-csa-cp8-off"],
                    stdout=log,
                    stderr=log,
                    check=True,
                )
            with (folder / "container_after.json").open("x") as snapshot:
                subprocess.run(
                    ["docker", "inspect", "dsv4-csa-cp8-off"],
                    stdout=snapshot,
                    check=True,
                )
            audit = audit_boundaries(folder, rows)
            (folder / "boundary_audit.json").write_text(json.dumps(audit, indent=2))
            result = json.loads((folder / "COMPLETE.json").read_text())
            if (
                result["successes"] != 15
                or result["failures"]
                or len(audit) != 15
                or not all(a["exact_boundary_verified"] for a in audit)
                or setting.read_text().strip() != host_value
            ):
                raise RuntimeError(
                    "Incomplete or mismatched serving batch; retain raw data"
                )
            summaries.append(
                dict(arm=arm, rate=rate, host_compaction=host_value, **result)
            )
            (args.output / "RESULTS.json").write_text(json.dumps(summaries, indent=2))
            prepared = folder / "prepared.json"
            print(f"COMPLETE_SERVING arm={arm} rate={rate} requests=15", flush=True)
        if setting.read_text().strip() != args.host_compaction:
            raise RuntimeError("Host policy changed")
    (args.output / "COMPLETE.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    # FLASH_SUPPLEMENT6_HOOK_V1072: user-requested targeted paired follow-up.
    import sys
    if ("--model-kind" in sys.argv and sys.argv[sys.argv.index("--model-kind")+1] == "pro"
            and "--output" in sys.argv
            and sys.argv[sys.argv.index("--output")+1] == "/tmp/ds_serial_retests_v1067/pro/internal/k_local"):
        subprocess.run(["python3", "/tmp/run_flash_supplement_v1072.py"], check=True)
    from internal_cohort_v1073 import pro_selected_arguments
    sys.argv = pro_selected_arguments(sys.argv)
    main()
