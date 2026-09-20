# SPDX-License-Identifier: Apache-2.0
"""Encode authorized public reconstruction on CPU, with explicit source aliases."""

# Standard
import argparse
from array import array
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

# Local
from audit_native_message_boundaries_v1024 import prefix_before_incoming_turn
from prepare_native_chain_replay_v587 import common_prefix
from rebuild_public_serving_v1045 import canonical_hash


def safe_source_id(dataset: str, original: str) -> str:
    """Create a cache-key-safe alias; retain the full source in the manifest."""
    labels = {"terminalbench_2.1": "tb21", "swe_bench_pro": "swepro"}
    return (
        "public-"
        + labels[dataset]
        + "-"
        + hashlib.sha256(original.encode()).hexdigest()
    )


def main() -> None:
    """Build both GLM token corpora without inference; keep ineligible rows visible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("/tmp/public_reconstructed_v1045")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--after-complete", type=Path)
    parser.add_argument("--before-formal", type=Path)
    args = parser.parse_args()
    if args.after_complete:
        deadline = time.monotonic() + 900
        while not args.after_complete.exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Selected serving rate did not complete")
            time.sleep(2)
    if args.before_formal and args.before_formal.exists():
        raise RuntimeError("Next formal serving phase already started; defer CPU work")
    args.output.mkdir(exist_ok=False)
    started = time.time()
    summaries = []
    with subprocess.Popen(
        [
            "docker",
            "exec",
            "-i",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "-e",
            "TOKENIZERS_PARALLELISM=false",
            "dsv4-csa-cp8-off",
            "python3",
            "/tmp/glm_native_tokenizer_v969.py",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as worker:
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("CPU tokenizer pipe missing")
        assets = json.loads(worker.stdout.readline())
        if not assets.get("ready"):
            raise RuntimeError(
                "Actual loaded checkpoint is not available to CPU tokenizer"
            )

        def encode(body: dict) -> list[int]:
            """Keep reconstructed content with the declared GLM template policy."""
            body = copy.deepcopy(body)
            body.setdefault("chat_template_kwargs", {})["clear_thinking"] = False
            worker.stdin.write(json.dumps(body) + "\n")
            worker.stdin.flush()
            result = json.loads(worker.stdout.readline())
            if "tokens" not in result:
                raise RuntimeError(f"Native encoder failed: {result}")
            return result["tokens"]

        try:
            for dataset in ("terminalbench_2.1", "swe_bench_pro"):
                source_manifest = args.source / dataset / "manifest.json"
                source = json.loads(source_manifest.read_text())
                folder = args.output / dataset
                folder.mkdir()
                packed = array("i")
                chains, audits, aliases = [], [], {}
                for item in source["requests"]:
                    record = json.loads(Path(item["reconstructed_file"]).read_text())
                    body = record["request_body"]
                    if (
                        record["replay_kind"] != "authorized_trajectory_reconstruction"
                        or canonical_hash(body) != item["body_sha256"]
                        or record["source_id"] != item["source_id"]
                        or record["source_usage"]["O"] != item["O"]
                    ):
                        raise ValueError("Reconstructed content/provenance changed")
                    tokens = encode(body)
                    prefix, count = prefix_before_incoming_turn(
                        body, include_response=False
                    )
                    history = common_prefix(tokens, encode(prefix))
                    alias = safe_source_id(dataset, item["source_id"])
                    if alias in aliases:
                        raise ValueError("Duplicate source alias")
                    aliases[alias] = item["source_id"]
                    seed_offset = len(packed)
                    packed.extend(tokens[:history])
                    offset = len(packed)
                    packed.extend(tokens)
                    aligned = history // 256 * 256
                    row = dict(
                        round=1,
                        source_request_id=alias,
                        original_source_id=item["source_id"],
                        source_file=item["reconstructed_file"],
                        source_line=1,
                        body_sha256=item["body_sha256"],
                        replay_kind="authorized_trajectory_reconstruction",
                        prompt_tokens=len(tokens),
                        token_offset=offset,
                        reusable_input_prefix_tokens=history,
                        new_input_tokens=len(tokens) - history,
                        output_budget=item["O"],
                        cluster=item["cluster"],
                        source_history=item["source_H"],
                        source_append=item["source_A"],
                        history_source="controlled_native_previous_prompt_LCP",
                        input_mode="reconstructed_native_content_prefix",
                        prefix_message_count=count,
                        expected_storage_aligned_history=aligned,
                        expected_storage_aligned_append=len(tokens) - aligned,
                        input_token_sha256=hashlib.sha256(
                            array("i", tokens).tobytes()
                        ).hexdigest(),
                    )
                    chains.append(
                        dict(
                            chain=len(chains),
                            first_timed_index=1,
                            rounds=[
                                dict(
                                    round=0,
                                    source_request_id="__native_prefix__:" + alias,
                                    prompt_tokens=history,
                                    token_offset=seed_offset,
                                    reusable_input_prefix_tokens=0,
                                    output_budget=4,
                                    setup_only=True,
                                ),
                                row,
                            ],
                        )
                    )
                    audit = dict(
                        case=item["case"],
                        **row,
                        native_bounds_ok=aligned >= 65536
                        and len(tokens) + item["O"] <= 524288,
                    )
                    audits.append(audit)
                    print(
                        json.dumps(
                            dict(
                                dataset=dataset,
                                case=item["case"],
                                H=aligned,
                                A=len(tokens) - aligned,
                                O=item["O"],
                                eligible=audit["native_bounds_ok"],
                            )
                        ),
                        flush=True,
                    )
                manifest = dict(
                    schema="public_glm_native_v1046",
                    model="deepseek-v4-pro",
                    model_display_name="GLM-5.2",
                    dataset=dataset,
                    replay_kind="authorized_trajectory_reconstruction",
                    selected_count=15,
                    measured_request_count=15,
                    selected_source_ids=list(aliases),
                    source_id_alias_map=aliases,
                    chains=chains,
                    unavailable=[],
                    replacements=[],
                    replacement_count=0,
                    continuation_of=None,
                    historical_reasoning_retained=True,
                    chat_template_override={"clear_thinking": False},
                    selection_sha256=hashlib.sha256(
                        source_manifest.read_bytes()
                    ).hexdigest(),
                    native_assets=assets,
                    provider_cache_boundary_reconstructed=False,
                    token_file_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
                    performance_metrics_used_for_selection=False,
                )
                with (folder / "tokens.i32le").open("xb") as stream:
                    packed.tofile(stream)
                (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
                (folder / "audit.json").write_text(json.dumps(audits, indent=2))
                summary = dict(
                    dataset=dataset,
                    count=len(audits),
                    eligible=sum(a["native_bounds_ok"] for a in audits),
                    min_H=min(a["expected_storage_aligned_history"] for a in audits),
                    max_H=max(a["expected_storage_aligned_history"] for a in audits),
                    inference_requests=0,
                )
                summaries.append(summary)
                (folder / "COMPLETE.json").write_text(
                    json.dumps(
                        dict(**summary, inputs_ready=summary["eligible"] == 15),
                        indent=2,
                    )
                )
        finally:
            worker.stdin.close()
            worker.wait(timeout=30)
    if worker.returncode:
        raise RuntimeError("Tokenizer worker exited unsuccessfully")
    (args.output / "SUMMARY.json").write_text(
        json.dumps(
            dict(
                datasets=summaries,
                start_wall=started,
                end_wall=time.time(),
                next_formal_started=bool(
                    args.before_formal and args.before_formal.exists()
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
