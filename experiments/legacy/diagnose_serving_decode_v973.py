# SPDX-License-Identifier: Apache-2.0
"""Replay one frozen request for isolated TPOT diagnostics, never formal serving."""

# Standard
from array import array
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import time
from typing import Any
import urllib.request

# Local
from prepare_serving_inputs_v909 import wait_seed_publication
from run_original_serving_v957 import formal_rows, idle_clear, verify
from ssd_file_readiness_v956 import wait_disk_publication
from serving_observation_v762 import StreamObservation
from strict_stream_completion_v909 import stream_completion


def assert_idle() -> None:
    """Reject a diagnostic request while the server has queued or running work."""
    with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=15) as response:
        values = [float(line.rsplit(" ", 1)[1])
                  for line in response.read().decode().splitlines()
                  if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{"))]
    if len(values) != 2 or any(values):
        raise RuntimeError("Diagnostic requires an idle server")


def control_nsys(session: str, action: str, output: Path) -> None:
    """Control an existing diagnostic session and retain its CLI output."""
    arguments = ["docker", "exec", "dsv4-csa-cp8-off",
                 "/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys",
                 action, "--session", session]
    if action == "start":
        arguments += ["--sample=none", "--cpuctxsw=none", "--output", str(output / "timeline")]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=600)
    (output / f"nsys_{action}.log").write_text(result.stdout + result.stderr)
    result.check_returncode()


def start_counter_sampler(output: Path, *, kernel_stacks: bool = False) -> list[tuple[subprocess.Popen, Any]]:
    """Count CPU cycles/instructions on all eight worker main threads for 5s.

    Counters start after the caller observes eight output events. This is not
    a GPU profile. The per-thread interval file retains scaling/availability
    information; unavailable counters must not be treated as zero work.
    """
    top = subprocess.check_output(
        ["docker", "top", "dsv4-csa-cp8-off", "-eo", "pid,args"], text=True
    )
    threads = {}
    for rank in range(8):
        matches = re.findall(
            rf"^\s*(\d+)\s+VLLM::Worker_TP{rank}_EP{rank}\s*$", top, re.M
        )
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify counter thread for rank {rank}")
        threads[rank] = int(matches[0])
    command = [
        "perf", "stat", "--no-inherit", "--per-thread",
        "-I", "500", "-x", ";", "-e", "{cycles,instructions,cycles:u,instructions:u}",
        "-e", "task-clock,context-switches,cpu-migrations",
        "-t", ",".join(str(pid) for pid in threads.values()),
        "-o", str(output / "main_thread_counters.txt"),
        "--", "sleep", "5",
    ]
    (output / "counter_contract.json").write_text(json.dumps(
        dict(threads=threads, command=command, duration_ms=5000,
             time_origin="perf start after eight client-visible events"), indent=2
    ))
    log = (output / "main_thread_counters.log").open("x")
    jobs = [(subprocess.Popen(command, stdout=log, stderr=log), log)]
    if kernel_stacks:
        compactors = []
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                name = (path / 'comm').read_text().strip()
            except OSError:
                continue
            if name.startswith('kcompactd'):
                compactors.append(int(path.name))
        targets = sorted(set(threads.values()) | set(compactors))
        kernel_command = [
            'perf', 'record', '-e', 'cycles:k', '-F', '199', '--call-graph', 'fp',
            '-t', ','.join(str(pid) for pid in targets),
            '-o', str(output / 'kernel_stacks.perf.data'), '--', 'sleep', '5',
        ]
        (output / 'kernel_contract.json').write_text(json.dumps(
            dict(threads=threads, compactors=compactors, command=kernel_command), indent=2))
        kernel_log = (output / 'kernel_stacks.log').open('x')
        jobs.append((subprocess.Popen(kernel_command, stdout=kernel_log, stderr=kernel_log), kernel_log))
    return jobs


def start_samplers(output: Path, kind: str = "python") -> list[tuple[subprocess.Popen, Any]]:
    """Sample four identified worker stacks for five seconds after prefill."""
    if kind in ("counters", "kernel_counters"):
        return start_counter_sampler(output, kernel_stacks=kind == "kernel_counters")
    top = subprocess.check_output(
        ["docker", "top", "dsv4-csa-cp8-off", "-eo", "pid,args"], text=True)
    jobs = []
    for rank in (0, 1, 3, 7):
        matches = re.findall(rf"^\s*(\d+)\s+VLLM::Worker_TP{rank}_EP{rank}\s*$", top, re.M)
        if len(matches) != 1:
            raise RuntimeError(f"Cannot identify rank {rank}")
        log = (output / f"rank{rank}.log").open("x")
        command = [
            "/tmp/serving_profiler_tools_v801/bin/py-spy", "record", "--pid", matches[0],
            "--duration", "5", "--rate", "100", "--nonblocking", "--threads", "--idle",
            "--full-filenames", "--format", "speedscope",
            "--output", str(output / f"rank{rank}.json")]
        if kind == "native":
            command = ["perf", "record", "-e", "cycles:u", "-F", "199",
                       "--call-graph", "dwarf,8192", "-t", matches[0],
                       "-o", str(output / f"rank{rank}.perf.data"), "--", "sleep", "5"]
        elif kind != "python":
            raise ValueError("Unsupported CPU sampler")
        process = subprocess.Popen(command, stdout=log, stderr=log)
        jobs.append((process, log))
    return jobs


def finish_samplers(jobs: list[tuple[subprocess.Popen, Any]]) -> list[str]:
    """Reap every diagnostic child, terminating stuck samplers without hiding failure."""
    errors = []
    for process, log in jobs:
        try:
            code = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
            errors.append(f"CPU sampler {process.pid} exceeded its completion deadline")
        else:
            if code:
                errors.append(f"CPU sampler {process.pid} exited {code}")
        finally:
            log.close()
    return errors


def run_request(body: dict[str, Any], row: dict[str, Any], output: Path,
                *, cpu: bool, nsys: str | None,
                dependency_log: bool = False,
                sample_slow_decode_ms: float | None = None,
                sampler_kind: str = "python",
                external_nsys_session: str | None = None) -> dict[str, Any]:
    """Run an unchanged request; preserve stream text, timing, and instrumentation."""
    assert_idle()
    if nsys and external_nsys_session:
        raise ValueError('Nsight session must have one controller')
    output.mkdir()
    encoded_body = json.dumps(body).encode()
    contract = {key:value for key,value in body.items() if key != 'prompt'}
    contract.update(prompt_tokens=len(body['prompt']),
                    prompt_i32_sha256=hashlib.sha256(array('i',body['prompt']).tobytes()).hexdigest(),
                    encoded_payload_sha256=hashlib.sha256(encoded_body).hexdigest())
    (output / 'request_contract.json').write_text(json.dumps(contract,indent=2))
    request = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
        data=encoded_body, headers={"Content-Type": "application/json"})
    samplers = []
    active = False
    captured = False
    sampler_start_event = None
    sampler_trigger_median_ms = None
    response_end = None
    text_parts = []
    # Start capture BEFORE sending work. Starting synchronously inside an SSE
    # read lets generation race ahead while the CLI starts, leaving only a
    # buffered stream to consume and a tiny/incomplete GPU capture.
    if nsys:
        control_nsys(nsys, "start", output)
        active = True
    observation = StreamObservation(time.monotonic())
    request_wall_start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1200) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    observation.done = True
                    break
                parsed = json.loads(data)
                observation.observe(parsed, time.monotonic())
                text_parts.extend(c.get("text", "") for c in parsed.get("choices", []))
                if observation.visible_events >= 8 and not captured:
                    if sample_slow_decode_ms is not None and not cpu:
                        recent = observation.visible_times[-8:]
                        median = statistics.median(1000*(b-a) for a,b in zip(recent,recent[1:]))
                        if median > sample_slow_decode_ms:
                            cpu = True
                            sampler_trigger_median_ms = median
                    captured = cpu or sample_slow_decode_ms is None
                    if cpu:
                        sampler_start_event = observation.visible_events
                        samplers = start_samplers(output, sampler_kind)
        response_end = time.monotonic()
        # Preserve completed stream evidence before the potentially long report
        # conversion. Profiler teardown is not request E2E latency.
        (output / 'stream_complete.json').write_text(json.dumps(dict(
            **observation.result(response_end, row['output_budget']),
            source_request_id=row['source_request_id'], input=row,
            nsys_session=nsys or external_nsys_session,
            formal_serving_result=False), indent=2))
        # Stop only after the full response. Stopping a busy GPU capture dropped
        # incomplete timestamp records, including rank0 model kernels.
        if active:
            time.sleep(2)
            active = False  # Stop requested; never issue it twice on a CLI timeout.
            control_nsys(nsys, "stop", output)
    finally:
        if active:
            control_nsys(nsys, "stop", output)
        (output / "output.txt").write_text("".join(text_parts))
        sampler_errors = finish_samplers(samplers)
    result = {**observation.result(response_end, row["output_budget"]),
              "source_request_id": row["source_request_id"], "input": row,
              "cpu_sampling": cpu, "nsys_session": nsys or external_nsys_session,
              "externally_controlled_nsys": external_nsys_session is not None,
              "cpu_sampler_kind": sampler_kind if cpu else None,
              "cpu_sampler_errors": sampler_errors,
              "cpu_sampling_complete": not sampler_errors if cpu else None,
              "sampler_start_visible_event": sampler_start_event,
              "sampler_trigger_median_ms": sampler_trigger_median_ms,
              "formal_serving_result": False,
              "output_text_sha256": hashlib.sha256("".join(text_parts).encode()).hexdigest()}
    (output / "request.json").write_text(json.dumps(result, indent=2))
    # Collect this request's dependency probes only after its latency is fixed.
    # Docker applies log-record wall timestamps, not a guessed boot-time offset.
    if dependency_log:
        with (output / 'dependency_request.log').open('x') as log:
            subprocess.run(['docker', 'logs', '--since', str(request_wall_start),
                            'dsv4-csa-cp8-off'], stdout=log, stderr=log, check=True)
    return result


def wait_diagnostic_seed(
    arm: str, request_id: str, prompt_tokens: int, started: float,
) -> dict:
    """Wait for the selected backend to publish an excluded seed request.

    Args:
        arm: k_local, no_prefetch, or ssd_only.
        request_id: Completed seed request identity.
        prompt_tokens: Seed length in the deployed model's tokens.
        started: Seed start wall-clock timestamp.
    Returns:
        Backend-specific publication evidence.
    Raises:
        ValueError: The comparison arm is unsupported.
        TimeoutError: The selected backend fails its publication deadline.
    """
    if arm == "ssd_only":
        return wait_disk_publication(request_id)
    if arm in ("k_local", "no_prefetch"):
        return wait_seed_publication(
            "dsv4-csa-cp8-off", request_id, prompt_tokens, started, chunk_size=256,
        )
    raise ValueError(f"Unsupported diagnostic arm: {arm}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse diagnostic CLI arguments; None reads the process command line.

    Return the parsed namespace. Raise SystemExit for invalid arguments or help.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("k_local", "no_prefetch", "ssd_only"), required=True)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--wait-complete", type=Path)
    parser.add_argument("--nsys-session")
    parser.add_argument("--prepare-seed", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--warm-and-gpu", action="store_true")
    parser.add_argument("--plain-only", action="store_true")
    parser.add_argument("--one-repeat", action="store_true")
    parser.add_argument("--warmup-skip-save", action="store_true",
                        help="Do not publish append KV during excluded warmup")
    parser.add_argument("--keep-seed-cache", action="store_true")
    parser.add_argument("--sample-slow-decode-ms", type=float)
    parser.add_argument("--native-cpu-sampling", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Wait for successful NP completion if requested, then isolate diagnostics."""
    args = parse_arguments()
    if args.wait_complete:
        while not args.wait_complete.exists():
            time.sleep(10)
        proof = json.loads(args.wait_complete.read_text())
        if (proof["successes"], proof["failures"], proof["request_records"]) != (29, 0, 29):
            raise RuntimeError("Formal NP batch did not succeed")
        # The parent coordinator is intentionally held before it can switch arms.
        # Freeze formal evidence BEFORE adding any diagnostic model requests.
        formal = args.wait_complete.parent
        with (formal / "server_after.log").open("x") as log:
            subprocess.run(["docker", "logs", "dsv4-csa-cp8-off"], stdout=log, stderr=log, check=True)
        with (formal / "container_after.json").open("x") as log:
            subprocess.run(["docker", "inspect", "dsv4-csa-cp8-off"], stdout=log, check=True)
    assert_idle()
    deployment = verify(args.arm, "serving_glm_configuration_v766", allow_diagnostic=True)
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    row = formal_rows(manifest)[args.case_index]
    tokens = array("i")
    with (args.inputs / "tokens.i32le").open("rb") as stream:
        stream.seek(row["token_offset"] * 4)
        tokens.fromfile(stream, row["prompt_tokens"])
    body = dict(model=manifest["model"], prompt=list(tokens), max_tokens=row["output_budget"],
                ignore_eos=True, temperature=0, stream=True,
                stream_options={"include_usage": True},
                kv_transfer_params={"lmcache.skip_save": False})
    args.output.mkdir()
    (args.output / "scope.json").write_text(json.dumps(dict(
        deployment=deployment, input=row, input_token_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
        first_request_warmup_excluded=True, formal_serving_result=False), indent=2))
    if args.prepare_seed:
        if args.keep_seed_cache:
            print("LEGACY_KEEP_SEED_CACHE_IGNORED_USER_REQUIRES_ISOLATED_CASES", flush=True)
        setup = {"clear": idle_clear(args.arm), "keep_existing_prefixes": False,
                 "legacy_keep_requested": args.keep_seed_cache}
        chain = manifest["chains"][args.case_index]
        if chain["first_timed_index"] != 1:
            raise RuntimeError("Selected diagnostic case has no explicit seed")
        seed_row = chain["rounds"][0]
        seed_tokens = array("i")
        with (args.inputs / "tokens.i32le").open("rb") as stream:
            stream.seek(seed_row["token_offset"] * 4)
            seed_tokens.fromfile(stream, seed_row["prompt_tokens"])
        started = time.time()
        result = stream_completion(list(seed_tokens), skip_save=False, model=manifest["model"])
        setup.update(input=seed_row, completion=asdict(result), readiness=wait_diagnostic_seed(
            args.arm, result.request_id, seed_row["prompt_tokens"], started))
        (args.output / "seed_excluded.json").write_text(json.dumps(setup, indent=2))
    with (args.output / "server_before.log").open("x") as log:
        subprocess.run(["docker", "logs", "dsv4-csa-cp8-off"], stdout=log, stderr=log, check=True)
    sequence = [("warmup", False, None), ("cpu", True, None)]
    if args.nsys_session:
        sequence.append(("gpu", False, args.nsys_session))
    sequence.extend((f"plain{repeat}", False, None) for repeat in range(3))
    if args.gpu_only:
        if not args.nsys_session:
            raise ValueError("GPU-only capture requires an Nsight session")
        sequence = [("gpu", False, args.nsys_session)]
    if args.warm_and_gpu:
        if not args.nsys_session or args.gpu_only or args.plain_only:
            raise ValueError('Warm-and-GPU requires a session and no other sequence mode')
        sequence = [('warmup', False, None), ('gpu', False, args.nsys_session)]
    if args.plain_only:
        if args.nsys_session or args.gpu_only:
            raise ValueError("Plain validation must not enable instrumentation")
        sequence = [("warmup", False, None)] + [
            (f"plain{repeat}", False, None) for repeat in range(1 if args.one_repeat else 3)]
    rows = []
    with (args.output / "gpu_telemetry.csv").open("x") as telemetry:
        monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu",
            "--format=csv", "-l", "1"], stdout=telemetry, stderr=telemetry)
        try:
            for label, cpu, nsys in sequence:
                request_body = body
                if label == 'warmup' and args.warmup_skip_save:
                    request_body = dict(body, kv_transfer_params={'lmcache.skip_save': True})
                result = run_request(request_body, row, args.output / label, cpu=cpu, nsys=nsys,
                    sample_slow_decode_ms=args.sample_slow_decode_ms,
                    sampler_kind="native" if args.native_cpu_sampling else "python",
                    dependency_log=(deployment.get('prediction_dependency_profiled', False)
                                    or deployment.get('service_wall_profiled', False)))
                result["cpu_retrieve_profiled"] = deployment.get("cpu_retrieve_profiled", False)
                result["pipeline_cpu_profiled"] = deployment.get("pipeline_cpu_profiled", False)
                result["capability_audit"] = deployment.get("capability_audit", False)
                result["backend_alias_fixed"] = deployment.get("backend_alias_fixed", False)
                result["prediction_dependency_mode"] = deployment.get("prediction_dependency_mode", "legacy")
                result["prediction_dependency_profiled"] = deployment.get("prediction_dependency_profiled", False)
                result["service_wall_profiled"] = deployment.get("service_wall_profiled", False)
                result["service_wall_light"] = deployment.get("service_wall_light", False)
                result["aligned_packet_candidate"] = deployment.get("aligned_packet_candidate", False)
                result["device_owner_candidate"] = deployment.get("device_owner_candidate", False)
                result["fixed_boundary_candidate"] = deployment.get("fixed_boundary_candidate", False)
                result["launch_queue_scale"] = deployment.get("launch_queue_scale", "1x")
                result["launch_queue_candidate"] = deployment.get("launch_queue_candidate", False)
                result["step_timeline_profiled"] = deployment.get("step_timeline_profiled", False)
                result["owner_read_workers"] = deployment.get("owner_read_workers", 1)
                result["owner_worker_candidate"] = deployment.get("owner_worker_candidate", False)
                result["stage_timing_profiled"] = deployment.get("stage_timing_profiled", False)
                result["prefetch_ctas_diagnostic"] = deployment.get("prefetch_ctas_diagnostic", 0)
                result["compact_width_diagnostic"] = deployment.get("compact_width_diagnostic")
                rows.append({"label": label, **result})
                print(json.dumps({"label": label, **{key: result[key] for key in
                    ("request_id", "ttft_ms", "tpot_ms", "output_text_sha256")}}), flush=True)
        finally:
            monitor.terminate()
            monitor.wait(timeout=15)
    with (args.output / "server_after.log").open("x") as log:
        subprocess.run(["docker", "logs", "dsv4-csa-cp8-off"], stdout=log, stderr=log, check=True)
    (args.output / "COMPLETE.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
