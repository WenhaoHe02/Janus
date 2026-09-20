# SPDX-License-Identifier: Apache-2.0
"""Compare the first three real cache reuses after private startup warmup."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

# Local
from startup_io import StartupClient
from warm_manifest import load_inputs


def main() -> None:
    """Record all three samples without saving append KV or claiming GPU timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-settle-seconds", type=float, default=0)
    args = parser.parse_args()
    ready = json.loads((args.run / "READY.json").read_text())
    plan = json.loads(args.plan.read_text())
    expected = dict(plan["env"], JANUS_ARTIFACT_ID=ready["artifact_id"])
    maximum = int(expected["LMCACHE_INDEXER_MAX_SEQ_LEN"])
    manifest, tokens, rows = load_inputs(args.inputs, maximum)
    warm_argv = plan["warmup"]
    index = warm_argv.index("--rank-endpoints") + 1
    client = StartupClient(
        plan["health_url"].removesuffix("/health"),
        warm_argv[index : index + 8],
        manifest["model"],
        args.run / "server.log",
    )
    client.verify_workers(expected)
    client.verify_loaded_native(Path(expected["JANUS_COUNTED_BUILD_RECEIPT"]))
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for number, row in enumerate(rows):
        start, count = row["token_offset"], row["prompt_tokens"]
        prompt = list(tokens[start : start + count])
        history = row["reusable_input_prefix_tokens"]
        if history < 256:
            raise ValueError("Reuse acceptance needs a storable real history")
        client.clear_dedicated_cache()
        tag = f"janus-reuse:{ready['artifact_id']}:{number}"
        seed = client.complete(prompt[:history], tag)
        client.wait_publication(seed["request_id"], history)
        if args.diagnostic_settle_seconds < 0:
            raise ValueError("Diagnostic settling time cannot be negative")
        time.sleep(args.diagnostic_settle_seconds)
        samples = []
        for repeat in range(3):
            log_start = (args.run / "server.log").stat().st_size
            sample = client.complete(prompt, tag, skip_save=True)
            sample["repeat"] = repeat + 1
            sample["restored_tokens"] = client.require_hit(
                sample["request_id"], history
            )
            client.drain()
            with (args.run / "server.log").open("rb") as stream:
                stream.seek(log_start)
                interval = stream.read().decode(errors="replace")
            (args.output / f"request_{number}_{repeat + 1}.log").write_text(interval)
            loads = re.findall(
                r"TUTTI_PROFILE batched_get keys=(\d+) loaded=(\d+) size_mb=([\d.]+)",
                interval,
            )
            sample["tutti_completed_load_batches"] = len(loads)
            sample["tutti_logged_load_mib_all_ranks"] = sum(float(x[2]) for x in loads)
            sample["tutti_load_evidence"] = len(loads) >= 8 and all(
                int(key) == int(loaded) and int(loaded) > 0 for key, loaded, _ in loads
            )
            sample["native_streaming_read_path"] = (
                "DSv4 streaming-only hit: skipped generic LMCacheEngine.retrieve"
                in interval
                and "compact native indexer stream started" in interval
                and "compact native indexer read plan" in interval
            )
            samples.append(sample)
            with (args.output / "samples.jsonl").open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            print(json.dumps(sample), flush=True)
        timings = [sample["ttft_ms"] for sample in samples]
        if any(value is None or value <= 0 for value in timings):
            raise ValueError("Cannot measure TTFT without nonempty token events")
        later = statistics.mean(timings[1:])
        results.append(
            dict(
                source_request_id=row["source_request_id"],
                history=history,
                append=count - history,
                samples=samples,
                first_over_later_mean=timings[0] / later,
                same_cache_hit=len({sample["restored_tokens"] for sample in samples})
                == 1,
                same_output=len({sample["output_sha256"] for sample in samples}) == 1,
            )
        )
    report = dict(
        ready=ready,
        diagnostic_settle_seconds=args.diagnostic_settle_seconds,
        results=results,
        plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        physical_io_verified=all(
            sample["tutti_load_evidence"] or sample["native_streaming_read_path"]
            for row in results
            for sample in row["samples"]
        ),
        note=(
            "Completed Tutti direct/streaming read paths confirm SSD participation. "
            "Byte totals do not fully account for device-counted sparse reads."
        ),
    )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
