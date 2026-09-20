# SPDX-License-Identifier: Apache-2.0
"""Prepare and time selected requests, treating missing predecessors as cold starts."""

# Standard
import argparse
from array import array
import asyncio
from dataclasses import asdict
import json
import importlib
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

# Local
from prepare_serving_inputs_v909 import wait_seed_publication
from run_agent_cases import clear_disk_cache
from run_glm52_recovery_smoke import token_sequence
from run_serving_periodic_v877 import complete
from strict_stream_completion_v909 import stream_completion
from ssd_file_readiness_v956 import wait_disk_publication
from serving_tutti_readiness_v986 import read_tutti_readiness
from serving_arrival_v1050 import dispatch_requests, use_serial_arrivals
from serving_serial_write_barrier_v1067 import (
    serial_write_barrier_enabled, wait_serial_write_boundary,
)


def formal_rows(manifest: dict) -> list[dict]:
    """Return one unchanged selected request per session, including cold first turns."""
    rows = []
    for chain in manifest["chains"]:
        first = chain["first_timed_index"]
        if first not in (0, 1) or len(chain["rounds"]) != first + 1:
            raise ValueError("Unexpected selected-session structure")
        rows.append(chain["rounds"][first])
    ids = [row["source_request_id"] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(
        manifest["selected_source_ids"]
    ):
        raise ValueError("Changed or duplicate selected source IDs")
    return rows


def get_json(url: str) -> dict:
    """Read a local serving endpoint without changing model state."""
    with urllib.request.urlopen(url, timeout=15) as response:
        return json.load(response)


def percentile(values: list[float], fraction: float) -> float | None:
    """Return a linearly interpolated percentile for non-empty values."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def idle_clear(arm: str) -> list[dict]:
    """Drain requests/writes then clear only this experiment's KV backend."""
    with urllib.request.urlopen(
        "http://127.0.0.1:8000/metrics", timeout=15
    ) as response:
        active = [
            float(line.rsplit(" ", 1)[1])
            for line in response.read().decode().splitlines()
            if line.startswith(
                ("vllm:num_requests_running{", "vllm:num_requests_waiting{")
            )
        ]
    if len(active) != 2 or any(active):
        raise RuntimeError("Refusing cache clear while requests are active")
    if arm == "ssd_only":
        wait_disk_publication()
    if arm in ("k_local", "no_prefetch"):
        deadline = time.monotonic() + 900
        while True:
            states = [
                get_json(f"http://127.0.0.1:{7000 + r}/write_slack/status")
                for r in range(8)
            ]
            if not any(
                s.get(k, 0)
                for s in states
                for k in ("queued_waves", "active_waves", "active_io_blocks")
            ):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Experimental writes did not drain")
            time.sleep(2)
    if arm != "dram":
        return clear_disk_cache()
    states = []
    for rank in range(8):
        req = urllib.request.Request(
            f"http://127.0.0.1:{7000 + rank}/cache/clear?locations=LocalCPUBackend",
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.load(response)
        if result.get("status") != "success":
            raise RuntimeError("Experimental CPU cache clear failed")
        states.append(result)
    return states


def verify(arm: str, configuration_module: str = "serving_ds_configuration_v890",
           *, allow_diagnostic: bool = False, allow_device_owner: bool = False) -> dict:
    """Verify worker controls and live Tutti readiness before submitting work.

    Args:
        arm: Requested comparison arm.
        configuration_module: Approved module defining the arm's runtime flags.
        allow_diagnostic: Allow explicit profiling or prefetch CTA experiments
            for diagnostic callers only. Formal serving must use False.
        allow_device_owner: Permit the explicitly selected GLM device-owner
            implementation for unprofiled serving; other diagnostics remain
            forbidden. This does not change prediction semantics or flags.
    Returns:
        Container identity, required flags, diagnostic marker and readiness proof.
    Raises:
        RuntimeError: Container/loader unavailable or profiling contaminates a
            formal serving deployment.
        ValueError: Configuration module or actual worker flags mismatch.
    """
    info = json.loads(
        subprocess.check_output(["docker", "inspect", "dsv4-csa-cp8-off"])
    )[0]
    if configuration_module not in (
        "serving_ds_configuration_v890", "serving_glm_configuration_v766"
    ):
        raise ValueError("Unsupported serving configuration module")
    flags = importlib.import_module(configuration_module).environment(arm)
    if not info["State"]["Running"]:
        raise RuntimeError("Experiment container is stopped")
    cpu_profiled = False
    pipeline_profiled = False
    capability_audit = False
    stage_timing_states: set[bool] = set()
    prefetch_cta_limits: set[int] = set()
    compact_width_states: set[str | None] = set()
    backend_alias_fixes: set[bool] = set()
    dependency_modes: set[str] = set()
    dependency_probes: set[bool] = set()
    service_wall_states: set[bool] = set()
    light_wall_states: set[bool] = set()
    aligned_packet_states: set[bool] = set()
    device_owner_states: set[bool] = set()
    fixed_boundary_states: set[bool] = set()
    launch_queue_scales: set[str] = set()
    launch_queue_markers: set[bool] = set()
    step_nvtx_states: set[bool] = set()
    owner_worker_states: set[int] = set()
    owner_worker_diagnostics: set[bool] = set()
    for rank in range(8):
        env = get_json(f"http://127.0.0.1:{7000 + rank}/env")
        step_nvtx = env.get('LMCACHE_SERVING_DIAGNOSTIC_STEP_NVTX') == '1'
        step_nvtx_states.add(step_nvtx)
        if step_nvtx:
            if not allow_diagnostic:
                raise RuntimeError('Step-NVTX profile cannot run formal serving')
            if not env.get('LMCACHE_SERVING_DIAGNOSTIC_NSYS'):
                raise ValueError('Step-NVTX profile has no Nsight session')
        queue_scale = env.get('CUDA_SCALE_LAUNCH_QUEUES', '1x')
        queue_marker = env.get('LMCACHE_SERVING_DIAGNOSTIC_LAUNCH_QUEUES')
        launch_queue_scales.add(queue_scale)
        launch_queue_markers.add(queue_marker is not None)
        if queue_marker is not None:
            if not allow_diagnostic:
                raise RuntimeError('Launch-queue candidate cannot run formal serving')
            if (arm != 'k_local' or queue_scale not in ('2x', '4x')
                    or queue_marker != queue_scale):
                raise ValueError('Launch-queue candidate configuration mismatch')
        elif queue_scale != '1x':
            raise ValueError('Unmarked launch-queue experiment')
        dependency_mode = env.get('LMCACHE_GLM_PREDICTION_IO_DEPENDENCY', 'legacy')
        dependency_profiled = env.get('LMCACHE_TUTTI_PREDICTION_DEPENDENCY_PROFILE') == '1'
        dependency_modes.add(dependency_mode)
        dependency_probes.add(dependency_profiled)
        if dependency_mode != 'legacy' or dependency_profiled:
            if not allow_diagnostic:
                raise RuntimeError('Prediction dependency diagnostic cannot run formal serving')
            if (dependency_mode not in ('capture', 'source') or arm != 'k_local'
                    or configuration_module != 'serving_glm_configuration_v766'):
                raise ValueError('Unsupported prediction dependency diagnostic')
            if (env.get('LMCACHE_GLM_DSA_OWNER_READ_WORKERS', '1') != '1'
                    or env.get('LMCACHE_GLM_DSA_ASYNC_PREDICTION', '0') != '0'):
                raise ValueError('Prediction dependency requires synchronous FIFO producers')
        backend_alias_fixes.add(env.get('LMCACHE_SERVING_FIX_BACKEND_ALIAS') == '1')
        capability_audit |= env.get('LMCACHE_SERVING_DIAGNOSTIC_CAPABILITY_AUDIT') == '1'
        if capability_audit and not allow_diagnostic:
            raise RuntimeError('Capability audit cannot run formal serving')
        cpu_profiled |= bool(env.get("LMCACHE_SERVING_DIAGNOSTIC_CPU_RETRIEVE_DIR"))
        pipeline_profiled |= bool(env.get('LMCACHE_SERVING_DIAGNOSTIC_PIPELINE_CPU_DIR'))
        if env.get('LMCACHE_SERVING_DIAGNOSTIC_PIPELINE_CPU_DIR') and not allow_diagnostic:
            raise RuntimeError('Pipeline CPU profiling cannot run formal serving')
        if env.get("LMCACHE_SERVING_DIAGNOSTIC_CPU_RETRIEVE_DIR") and not allow_diagnostic:
            raise RuntimeError("CPU-profiled deployment cannot run formal serving")
        stage_timing = env.get("LMCACHE_SERVING_DIAGNOSTIC_STAGE_TIMING") == '1'
        stage_timing_states.add(stage_timing)
        if stage_timing and not allow_diagnostic:
            raise RuntimeError("Stage-timed deployment cannot run formal serving")
        cta_limit = int(env.get('LMCACHE_SERVING_DIAGNOSTIC_PREFETCH_CTAS', '0'))
        prefetch_cta_limits.add(cta_limit)
        if cta_limit:
            if not allow_diagnostic:
                raise RuntimeError('CTA diagnostic deployment cannot run formal serving')
            if cta_limit < 1 or arm != 'k_local':
                raise ValueError('Invalid prefetch CTA diagnostic')
            if (env.get('LMCACHE_SSD_TP_GATHER_MAX_CTAS') != str(cta_limit)
                    or env.get('LMCACHE_SSD_TP_GATHER_MAX_NCHANNELS') != '0'):
                raise ValueError('Prefetch CTA controls mismatch')
        expected = dict(flags)
        owner_workers = int(env.get('LMCACHE_GLM_DSA_OWNER_READ_WORKERS', '1'))
        worker_diagnostic = env.get('LMCACHE_SERVING_DIAGNOSTIC_OWNER_WORKERS')
        owner_worker_states.add(owner_workers)
        owner_worker_diagnostics.add(worker_diagnostic is not None)
        if worker_diagnostic is not None:
            if not allow_diagnostic:
                raise RuntimeError('Owner-worker candidate cannot run formal serving')
            if (arm != 'k_local' or configuration_module != 'serving_glm_configuration_v766'
                    or owner_workers not in (2, 4) or int(worker_diagnostic) != owner_workers):
                raise ValueError('Owner-worker diagnostic mismatch')
        elif configuration_module == 'serving_glm_configuration_v766' and owner_workers != 1:
            raise ValueError('Unmarked owner-worker experiment')
        aligned_packet = env.get('LMCACHE_SERVING_DIAGNOSTIC_ALIGNED_PACKET') == '1'
        aligned_packet_states.add(aligned_packet)
        if aligned_packet:
            if not allow_diagnostic:
                raise RuntimeError('Aligned-packet candidate cannot run formal serving')
            if arm != 'k_local' or env.get('LMCACHE_CSA_OWNER_KV_PACKET') != '1':
                raise ValueError('Aligned packet configuration mismatch')
        elif env.get('LMCACHE_CSA_OWNER_KV_PACKET', '0') != '0' and not (
                configuration_module == 'serving_ds_configuration_v890'
                and flags.get('LMCACHE_CSA_OWNER_KV_PACKET') == '1'
                and env.get('LMCACHE_DSV4_DEVICE_OWNER_READ') == '1'):
            raise ValueError('Unmarked owner-packet experiment')
        service_wall = bool(env.get('LMCACHE_SERVING_DIAGNOSTIC_SERVICE_WALL_DIR'))
        fixed_boundary = env.get('LMCACHE_SERVING_DIAGNOSTIC_FIXED_BOUNDARY') == '1'
        fixed_boundary_states.add(fixed_boundary)
        if fixed_boundary:
            if not allow_diagnostic:
                raise RuntimeError('Fixed-boundary candidate cannot run formal serving')
            if (arm != 'k_local' or configuration_module != 'serving_glm_configuration_v766'
                    or env.get('LMCACHE_GLM_DSA_FIXED_BOUNDARY_GATHER') != '1'
                    or env.get('LMCACHE_GLM_DSA_DEVICE_OWNER_READ') != '1'
                    or not aligned_packet):
                raise ValueError('Fixed-boundary candidate configuration mismatch')
        elif env.get('LMCACHE_GLM_DSA_FIXED_BOUNDARY_GATHER','0') != '0':
            raise ValueError('Unmarked fixed-boundary candidate')
        device_owner = env.get('LMCACHE_SERVING_DIAGNOSTIC_DEVICE_OWNER') == '1'
        device_owner_states.add(device_owner)
        if device_owner:
            if not allow_diagnostic and not allow_device_owner:
                raise RuntimeError('Device-owner candidate cannot run formal serving')
            if (arm != 'k_local' or configuration_module != 'serving_glm_configuration_v766'
                    or env.get('LMCACHE_GLM_DSA_DEVICE_OWNER_READ') != '1'):
                raise ValueError('Device-owner candidate configuration mismatch')
        elif env.get('LMCACHE_GLM_DSA_DEVICE_OWNER_READ', '0') != '0':
            raise ValueError('Unmarked device-owner candidate')
        light_wall = env.get('LMCACHE_SERVING_DIAGNOSTIC_LIGHT_SERVICE_WALL') == '1'
        light_wall_states.add(light_wall)
        if light_wall and not service_wall:
            raise ValueError('Light service marker without its recorder')
        service_wall_states.add(service_wall)
        if service_wall:
            if not allow_diagnostic:
                raise RuntimeError('Service-wall diagnostic cannot run formal serving')
            if arm != 'k_local' or configuration_module != 'serving_glm_configuration_v766':
                raise ValueError('Unsupported service-wall diagnostic')
            expected['LMCACHE_TUTTI_PROFILE'] = '0' if light_wall else '1'
        compact_diagnostic = env.get('LMCACHE_SERVING_DIAGNOSTIC_COMPACT_WIDTH')
        compact_width_states.add(compact_diagnostic)
        if compact_diagnostic is not None:
            if not allow_diagnostic:
                raise RuntimeError('Compact-width diagnostic cannot run formal serving')
            if compact_diagnostic not in ('0', '1') or arm != 'k_local':
                raise ValueError('Invalid compact-width diagnostic')
            expected['LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH'] = compact_diagnostic
        if stage_timing:
            if configuration_module != "serving_glm_configuration_v766":
                raise ValueError("Host stage diagnostic currently supports GLM only")
            expected['LMCACHE_TTFT_STAGE_PROFILE'] = '1'
        differences = {k: env.get(k) for k, v in expected.items() if env.get(k) != v}
        if differences:
            raise ValueError(f"rank {rank} differs: {differences}")
    if len(stage_timing_states) != 1:
        raise ValueError("Host stage diagnostic differs across ranks")
    if len(prefetch_cta_limits) != 1:
        raise ValueError('Prefetch CTA controls differ across ranks')
    if len(compact_width_states) != 1:
        raise ValueError('Compact-width diagnostic differs across ranks')
    if len(backend_alias_fixes) != 1:
        raise ValueError('Backend alias fix differs across ranks')
    if len(dependency_modes) != 1 or len(dependency_probes) != 1:
        raise ValueError('Prediction dependency controls differ across ranks')
    if len(service_wall_states) != 1:
        raise ValueError('Service-wall diagnostic differs across ranks')
    if len(light_wall_states) != 1:
        raise ValueError('Light service-wall controls differ across ranks')
    if len(aligned_packet_states) != 1:
        raise ValueError('Aligned-packet candidate differs across ranks')
    if len(device_owner_states) != 1:
        raise ValueError('Device-owner candidate differs across ranks')
    if len(fixed_boundary_states) != 1:
        raise ValueError('Fixed-boundary candidate differs across ranks')
    if len(launch_queue_scales) != 1 or len(launch_queue_markers) != 1:
        raise ValueError('Launch-queue candidate differs across ranks')
    if len(step_nvtx_states) != 1:
        raise ValueError('Step-NVTX profile differs across ranks')
    if len(owner_worker_states) != 1 or len(owner_worker_diagnostics) != 1:
        raise ValueError('Owner-worker controls differ across ranks')
    compact_diagnostic = next(iter(compact_width_states))
    if compact_diagnostic is not None:
        flags['LMCACHE_CSA_OWNER_GPU_COMPACT_WIDTH'] = compact_diagnostic
    if True in stage_timing_states:
        flags['LMCACHE_TTFT_STAGE_PROFILE'] = '1'
    tutti_readiness = read_tutti_readiness(arm)
    return dict(container_id=info["Id"], required_worker_env=flags,
                configuration_module=configuration_module,
                cpu_retrieve_profiled=cpu_profiled,
                pipeline_cpu_profiled=pipeline_profiled,
                capability_audit=capability_audit,
                backend_alias_fixed=True in backend_alias_fixes,
                prediction_dependency_mode=next(iter(dependency_modes)),
                prediction_dependency_profiled=True in dependency_probes,
                service_wall_profiled=True in service_wall_states,
                service_wall_light=True in light_wall_states,
                aligned_packet_candidate=True in aligned_packet_states,
                device_owner_candidate=True in device_owner_states,
                fixed_boundary_candidate=True in fixed_boundary_states,
                launch_queue_scale=next(iter(launch_queue_scales)),
                launch_queue_candidate=True in launch_queue_markers,
                step_timeline_profiled=True in step_nvtx_states,
                owner_read_workers=next(iter(owner_worker_states)),
                owner_worker_candidate=True in owner_worker_diagnostics,
                stage_timing_profiled=True in stage_timing_states,
                prefetch_ctas_diagnostic=next(iter(prefetch_cta_limits)),
                compact_width_diagnostic=compact_diagnostic,
                tutti_readiness=tutti_readiness)


def prepare_run(args: argparse.Namespace, manifest: dict, tokens: array) -> dict:
    """Exclude compilation/setup and preserve each manifest's explicit seed policy."""
    proof = verify(args.arm, args.configuration_module,
                   allow_device_owner=getattr(args, 'device_owner', False))
    served_models = [row["id"] for row in get_json("http://127.0.0.1:8000/v1/models")["data"]]
    if manifest["model"] not in served_models:
        raise ValueError("Input manifest model differs from the loaded server")
    args.output.mkdir(exist_ok=False)
    with (args.output / "preparation.jsonl").open("x") as log:

        def record(row: dict) -> None:
            """Persist completed setup stages, separate from formal timing."""
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)

        record(
            dict(phase="cache_clear_before_preparation", results=idle_clear(args.arm))
        )
        actual_warmup = manifest.get('native_content_warmup') is True
        if args.reuse_shapes:
            prior = json.loads(args.reuse_shapes.read_text())
            if (
                prior["container_id"] != proof["container_id"]
                or prior["manifest_sha256"] != manifest["selection_sha256"]
                or (actual_warmup and prior.get('warmup_input_mode') != 'actual_native_content')
            ):
                raise ValueError(
                    "Cannot reuse compilation from different container/cohort"
                )
            record(dict(phase="compiled_shapes_reused", proof=str(args.reuse_shapes)))
        elif actual_warmup:
            from warmup_actual_requests_v1073 import warm_actual_requests

            warm_actual_requests(formal_rows(manifest), tokens, manifest['model'], args.arm,
                                 record, stream_completion, wait_seed_publication,
                                 wait_disk_publication)
            record(dict(phase='actual_warmup_cache_cleared', results=idle_clear(args.arm)))
        else:
            shapes = sorted(
                {
                    (
                        r["reusable_input_prefix_tokens"] // 256 * 256,
                        r["prompt_tokens"]
                        - r["reusable_input_prefix_tokens"] // 256 * 256,
                    )
                    for r in formal_rows(manifest)
                }
            )
            for index, (history, append) in enumerate(shapes):
                prefix = token_sequence(history, 931000 + index) if history else []
                if history:
                    started = time.time()
                    result = stream_completion(prefix, skip_save=False, model=manifest["model"])
                    if args.arm == "ssd_only":
                        readiness = wait_disk_publication(result.request_id)
                        record(dict(phase="synthetic_seed_ready", readiness=readiness))
                    if args.arm in ("k_local", "no_prefetch"):
                        wait_seed_publication(
                            "dsv4-csa-cp8-off",
                            result.request_id,
                            history,
                            started,
                            chunk_size=256,
                        )
                prompt = prefix + token_sequence(append, 932000 + index)
                for repeat in (1,):
                    result = stream_completion(prompt, skip_save=True, model=manifest["model"])
                    record(
                        dict(
                            phase="warmup_excluded",
                            H=history,
                            A=append,
                            repeat=repeat,
                            completion=asdict(result),
                        )
                    )
            record(dict(phase="synthetic_cache_cleared", results=idle_clear(args.arm)))
        for chain in manifest["chains"]:
            if chain["first_timed_index"] == 0:
                continue
            row = chain["rounds"][0]
            start = row["token_offset"]
            started = time.time()
            result = stream_completion(
                list(tokens[start : start + row["prompt_tokens"]]), skip_save=False,
                model=manifest["model"],
                cache_tag=(chain['rounds'][chain['first_timed_index']]['source_request_id']
                           if getattr(args, 'isolate_request_cache', False) else None),
            )
            readiness = None
            if args.arm == "ssd_only":
                readiness = wait_disk_publication(result.request_id)
            if args.arm in ("k_local", "no_prefetch"):
                readiness = wait_seed_publication(
                    "dsv4-csa-cp8-off",
                    result.request_id,
                    row["prompt_tokens"],
                    started,
                    chunk_size=256,
                )
            record(
                dict(
                    phase="original_predecessor_setup_excluded",
                    source_request_id=row["source_request_id"],
                    completion=asdict(result),
                    readiness=readiness,
                )
            )
    proof.update(
        warmup_input_mode=('actual_native_content' if actual_warmup else 'synthetic_shape'),
        manifest_sha256=manifest["selection_sha256"],
        arm=args.arm,
        missing_predecessor=manifest.get("missing_predecessor_policy"),
        profile=False,
        independent_request_cache=getattr(args, 'isolate_request_cache', False),
        continuation_of=manifest.get("continuation_of"),
        replacement_count=manifest.get("replacement_count", 0),
    )
    (args.output / "prepared.json").write_text(json.dumps(proof, indent=2))
    return proof


async def replay(
    args: argparse.Namespace, manifest: dict, tokens: array, proof: dict
) -> None:
    """Run unchanged requests with explicitly recorded periodic/serial arrivals."""
    if verify(args.arm, args.configuration_module,
              allow_device_owner=getattr(args, 'device_owner', False))["container_id"] != proof["container_id"]:
        raise ValueError("Container changed after setup")
    rows = formal_rows(manifest)
    serial = args.prefill_only or getattr(args, "serial", False)
    write_boundary = serial_write_barrier_enabled(serial, args.configuration_module)
    start = time.monotonic()
    finished = []
    (args.output / "runtime.json").write_text(
        json.dumps(
            dict(
                inputs=str(args.inputs),
                arm=args.arm,
                rate=None if serial else args.rate,
                requested_rate_slot=args.rate,
                arrival_mode="serial" if serial else "periodic",
                serial_completion_boundary=("response_and_write_publication_v1067"
                                            if write_boundary else "response_only"),
                arrival_policy="user_0p005_to_serial_v1050" if getattr(args, "serial", False) else None,
                source_ids=manifest["selected_source_ids"],
                selected_count=manifest["selected_count"],
                runnable=len(rows),
                missing_predecessor=manifest.get("missing_predecessor_policy"),
                replacements=manifest.get("replacements", []),
                continuation_of=manifest.get("continuation_of"),
            continuation_note=(
                "Separate arrival segment after preparation; not uninterrupted replay"
            )
                if manifest.get("continuation_of")
                else None,
                decode_mode="normal_demand_decode",
                source_output_preserved=True,
                prefill_only=args.prefill_only,
                independent_request_cache=getattr(args, 'isolate_request_cache', False),
                prefill_metric="mean(prompt_tokens / (ttft_ms/1000))",
                proof=proof,
            ),
            indent=2,
        )
    )
    with (args.output / "requests.jsonl").open("x") as log:

        def record(row: dict) -> None:
            """Retain failures and successful timings without replacing traces."""
            finished.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in ("source_request_id", "status", "ttft_ms", "tpot_ms")
                    }
                ),
                flush=True,
            )

        for unavailable in manifest["unavailable"]:
            record(unavailable)

        async def session(index: int, row: dict) -> None:
            due = time.monotonic() if serial else start + index / args.rate
            await asyncio.sleep(max(0, due - time.monotonic()))
            offset = row["token_offset"]
            prompt = list(tokens[offset : offset + row["prompt_tokens"]])
            served_budget = 1 if args.prefill_only else row["output_budget"]
            started_wall = time.time()
            result = await asyncio.to_thread(
                complete,
                "http://127.0.0.1:8000",
                manifest["model"],
                prompt,
                served_budget,
                cache_tag=(row['source_request_id'] if getattr(args, 'isolate_request_cache', False) else None),
            )
            record(
                dict(
                    source_request_id=row["source_request_id"],
                    source_prompt_tokens=row["prompt_tokens"],
                    source_output=row["output_budget"],
                    served_output=served_budget,
                    source_history=row.get("source_history"),
                    source_append=row.get("source_append"),
                    replay_history=row["reusable_input_prefix_tokens"],
                    replay_append=row["prompt_tokens"]
                    - row["reusable_input_prefix_tokens"],
                    history_source=row.get("history_source"),
                    prefill_only=args.prefill_only,
                    cluster=row["cluster"],
                    input_mode=row["input_mode"],
                    scheduled_session_s=due,
                    chain=index,
                    source_round=row["round"],
                    tool_delay_s=0,
                    cache_tag=(row['source_request_id'] if getattr(args, 'isolate_request_cache', False) else None),
                    **result,
                )
            )
            if write_boundary:
                barrier = await asyncio.to_thread(
                    wait_serial_write_boundary, args.arm, proof["container_id"],
                    result, len(prompt), started_wall,
                    publication=wait_seed_publication,
                    disk_publication=wait_disk_publication, rank_status=get_json,
                )
                barrier["source_request_id"] = row["source_request_id"]
                with (args.output / "write_barriers.jsonl").open("a") as evidence:
                    evidence.write(json.dumps(barrier) + "\n")
                print(json.dumps({"WRITE_BOUNDARY": barrier}), flush=True)

        await dispatch_requests(rows, session, serial)
    success = [r for r in finished if r["status"] == "success"]
    aggregate = {}
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [r[key] for r in success if r.get(key) is not None]
        aggregate[key] = dict(
            n=len(values),
            mean=statistics.mean(values) if values else None,
            p50=percentile(values, 0.50),
            p95=percentile(values, 0.95),
            p99=percentile(values, 0.99),
            population_std=statistics.pstdev(values) if values else None,
        )
    values = [
        r["usage"]["prompt_tokens"] / (r["ttft_ms"] / 1000)
        for r in success
        if r.get("ttft_ms", 0) > 0
    ]
    aggregate["prefill_tokens_per_s"] = dict(
        mean=statistics.mean(values) if values else None,
        n=len(values),
        definition=(
            "arithmetic mean of per-request prompt_tokens / client TTFT seconds; "
            "includes cache reuse and queueing"
        ),
    )
    result = dict(
        selected=manifest["selected_count"],
        request_records=len(finished),
        successes=len(success),
        failures=sum(r["status"] == "failed" for r in finished),
        unavailable=len(manifest["unavailable"]),
        aggregate=aggregate,
        timed_interval_s=time.monotonic() - start,
        arrival_mode="serial" if serial else "periodic",
        actual_arrival_rate=None if serial else args.rate,
        requested_rate_slot=args.rate,
    )
    (args.output / "COMPLETE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main() -> None:
    """Prepare then measure one arm/rate using immutable selected native inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=["k_local", "no_prefetch", "ssd_only", "dram"], required=True
    )
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--configuration-module", default="serving_ds_configuration_v890",
                        choices=("serving_ds_configuration_v890", "serving_glm_configuration_v766"))
    parser.add_argument("--reuse-shapes", type=Path)
    parser.add_argument("--device-owner", action="store_true",
                        help="Use the verified GLM device-owner K implementation without enabling profiling")
    parser.add_argument("--isolate-request-cache", action="store_true",
                        help="Use LMCache request tags so future seeds cannot pre-cache another request's append")
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="serve one output token while preserving source output metadata",
    )
    args = parser.parse_args()
    args.serial = use_serial_arrivals(
        args.rate, args.output, Path(__file__).with_name("serving_arrival_policy_v1050.json")
    )
    if args.serial:
        print("ARRIVAL_MODE serial; replaces requested0.005 slot, actual rate=None; full O preserved", flush=True)
    if args.device_owner and (args.arm != 'k_local' or args.configuration_module != 'serving_glm_configuration_v766'):
        raise ValueError('Device-owner serving applies only to the GLM K arm')
    if args.rate <= 0:
        raise ValueError("Rate must be positive")
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    tokens = array("i")
    path = args.inputs / "tokens.i32le"
    with path.open("rb") as source:
        tokens.fromfile(source, path.stat().st_size // 4)
    proof = prepare_run(args, manifest, tokens)
    asyncio.run(replay(args, manifest, tokens, proof))


if __name__ == "__main__":
    main()
