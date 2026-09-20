# SPDX-License-Identifier: Apache-2.0
"""Capture CPU hardware counters on one already prepared long original request."""

# Standard
from array import array
import argparse
import json
import os
from pathlib import Path
import time

# Local
from diagnose_serving_decode_v973 import assert_idle, run_request
from run_original_serving_v957 import formal_rows, verify


def task_snapshot() -> dict:
    """Read host thread CPU counters and names only, without command arguments."""
    rows = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            tasks = list((process / "task").iterdir())
        except OSError:
            continue
        for task in tasks:
            try:
                raw = (task / "stat").read_text()
            except OSError:
                continue
            name = raw[raw.index("(") + 1 : raw.rindex(")")]
            fields = raw.rsplit(")", 1)[1].split()
            rows[f"{process.name}/{task.name}"] = dict(
                pid=int(process.name),
                tid=int(task.name),
                name=name,
                user_ticks=int(fields[11]),
                system_ticks=int(fields[12]),
                start_ticks=fields[19],
                last_cpu=int(fields[36]),
            )
    vmstat = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        name, value = line.split()
        if name.startswith(
            ("compact_", "pgmigrate", "pgfault", "pgmajfault", "thp_", "numa_")
        ):
            vmstat[name] = int(value)
    return dict(
        monotonic_s=time.monotonic(),
        threads=rows,
        vmstat=vmstat,
        compaction_proactiveness=Path("/proc/sys/vm/compaction_proactiveness")
        .read_text()
        .strip(),
        meminfo=Path("/proc/meminfo").read_text(),
    )


def main() -> None:
    """Repeat the unchanged native109K case once with decode-only CPU counters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-cpu-census", action="store_true")
    parser.add_argument("--kernel-stacks", action="store_true")
    args = parser.parse_args()
    assert_idle()
    proof = verify("k_local", "serving_glm_configuration_v766", allow_diagnostic=True)
    if (
        proof["container_id"]
        != "f1011ede1a5fd1934bcfc75928d5f56316a9c58a03105be94ef7aa96ce79348f"
    ):
        raise RuntimeError("Expected unchanged normal GLM deployment")
    manifest = json.loads(
        Path("/tmp/glm_long_native_inputs_v1025/manifest.json").read_text()
    )
    row = formal_rows(manifest)[0]
    tokens = array("i")
    with Path("/tmp/glm_long_native_inputs_v1025/tokens.i32le").open("rb") as source:
        source.seek(row["token_offset"] * 4)
        tokens.fromfile(source, row["prompt_tokens"])
    body = dict(
        model=manifest["model"],
        prompt=list(tokens),
        max_tokens=row["output_budget"],
        ignore_eos=True,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        kv_transfer_params={"lmcache.skip_save": False},
    )
    before = task_snapshot() if args.host_cpu_census else None
    result = run_request(
        body,
        row,
        args.output,
        cpu=True,
        nsys=None,
        sampler_kind="kernel_counters" if args.kernel_stacks else "counters",
    )
    if before is not None:
        after = task_snapshot()
        elapsed = after["monotonic_s"] - before["monotonic_s"]
        hz = os.sysconf("SC_CLK_TCK")
        deltas = []
        for key, current in after["threads"].items():
            previous = before["threads"].get(key)
            if previous is None or current["start_ticks"] != previous["start_ticks"]:
                continue
            ticks = (
                current["user_ticks"]
                + current["system_ticks"]
                - previous["user_ticks"]
                - previous["system_ticks"]
            )
            if ticks > 0:
                deltas.append(
                    dict(current, cpu_s=ticks / hz, cpu_fraction=ticks / hz / elapsed)
                )
        deltas.sort(key=lambda r: -r["cpu_s"])
        (args.output / "host_cpu_census.json").write_text(
            json.dumps(
                dict(
                    before=before,
                    after=after,
                    measured_elapsed_s=elapsed,
                    deltas=deltas,
                    request_id=result["request_id"],
                ),
                indent=2,
            )
        )
        print(json.dumps(dict(top_host_threads=deltas[:14])), flush=True)
    print(
        json.dumps(
            dict(
                request_id=result["request_id"],
                ttft_ms=result["ttft_ms"],
                tpot_ms=result["tpot_ms"],
                counters_complete=result["cpu_sampling_complete"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
