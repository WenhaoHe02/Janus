# SPDX-License-Identifier: Apache-2.0
"""Re-encode the fixed internal15 from source content with a loaded DS server."""

# Standard
import argparse
from array import array
from collections.abc import Callable
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

# Local
from audit_native_message_boundaries_v1024 import prefix_before_incoming_turn
from build_existing29_model_inputs_v957 import body_hash
from prepare_native_chain_replay_v587 import common_prefix, native_tokens
from build_public_glm_native_v1046 import safe_source_id
from rebuild_public_serving_v1045 import canonical_hash


def validate_checkpoint_mount(info: dict, checkpoint: Path, model_kind: str) -> dict:
    """Validate the running read-only checkpoint bind, not unavailable file hashes.

    Args:
        info: Actual Docker inspection record.
        checkpoint: Requested checkpoint directory.
        model_kind: Expected DS variant.
    Returns:
        Matching read-only bind identity.
    Raises:
        ValueError: Wrong container state, source path or model variant.
    """
    expected = {"pro": "DeepSeek-V4-Pro-HF64", "flash": "DeepSeek-V4-Flash-HF"}
    mounts = [m for m in info["Mounts"] if m["Destination"] == "/pro_model"]
    if (
        not info["State"]["Running"]
        or len(mounts) != 1
        or checkpoint.name != expected[model_kind]
        or mounts[0]["Source"] != str(checkpoint)
        or mounts[0]["RW"] is not False
    ):
        raise ValueError("Running checkpoint bind differs from requested model")
    return mounts[0]


def preserve_ds_history(body: dict) -> dict:
    """Normalize the deprecated reasoning field without dropping source content.

    Args:
        body: Original request body; never mutated.
    Returns:
        Native-render request preserving each original reasoning string via the
        DS renderer's supported `reasoning` field and drop_thinking=False.
    Raises:
        ValueError: Conflicting reasoning aliases or an incompatible render mode.
    """
    request = copy.deepcopy(body)
    has_reasoning = False
    for message in request["messages"]:
        original = message.get("reasoning_content")
        current = message.get("reasoning")
        if original and current and original != current:
            raise ValueError("Conflicting source reasoning fields")
        if original:
            message["reasoning"] = original
        has_reasoning |= bool(message.get("reasoning"))
    options = request.setdefault("chat_template_kwargs", {})
    options["drop_thinking"] = False
    if has_reasoning:
        if options.get("reasoning_effort") == "none":
            raise ValueError("Source reasoning conflicts with chat-only render mode")
        options["enable_thinking"] = True
    return request


def encode_reconstructed(
    source: dict, selected: dict, encode: Callable[[dict], list[int]], max_total: int
) -> tuple[list[int], int, int]:
    """Encode an authorized public replay without claiming original HTTP fidelity.

    Args:
        source: Labeled reconstructed body and original usage/provenance.
        selected: Frozen reconstructed-manifest row with original ID, hash and O.
        encode: Actual DS native tokenizer, never a different model's token file.
        max_total: Native prompt plus output capacity.
    Returns:
        Full token IDs, native historical LCP and prefix message count.
    Raises:
        ValueError: Reconstruction provenance or native length is invalid.
    """
    body = source["request_body"]
    if (
        source.get("replay_kind") != "authorized_trajectory_reconstruction"
        or source.get("source_id") != selected["source_id"]
        or source.get("source_usage", {}).get("O") != selected["O"]
        or canonical_hash(body) != selected["body_sha256"]
    ):
        raise ValueError("Frozen public reconstruction identity/content/output changed")
    tokens = encode(copy.deepcopy(body))
    prefix, count = prefix_before_incoming_turn(body, include_response=False)
    history = common_prefix(tokens, encode(copy.deepcopy(prefix)))
    if history // 256 * 256 < 65536 or len(tokens) + selected["O"] > max_total:
        raise ValueError(
            "Native public input violates64K/capacity; no silent replacement"
        )
    return tokens, history, count


def encode_original(
    source: dict, selected: dict, encode: Callable[[dict], list[int]], max_total: int
) -> tuple[list[int], int, int]:
    """Encode original content and its historical prompt; never transfer provider H.

    Args:
        source: Original successful request/response record.
        selected: Frozen source identity, output budget, and content hash.
        encode: Native DS tokenizer accepting an unchanged request body.
        max_total: Capacity including the original output budget.
    Returns:
        Complete token IDs, exact common-prefix length, prefix message count.
    Raises:
        ValueError: Identity/content/output changed or native length is invalid.
    """
    body = source["request_body"]
    location = selected.get("original", selected)
    if (
        source["request_id"] != selected["source_id"]
        or source["response_status"] != 200
        or source["completion_tokens"] != selected["O"]
        or body_hash(body) != location["body_sha256"]
    ):
        raise ValueError("Frozen source identity/content/output changed")
    tokens = encode(copy.deepcopy(body))
    prefix, count = prefix_before_incoming_turn(body, include_response=False)
    history = common_prefix(tokens, encode(copy.deepcopy(prefix)))
    if history // 256 * 256 < 65536 or len(tokens) + selected["O"] > max_total:
        raise ValueError("Native input violates64K history/capacity; do not replace")
    return tokens, history, count


def main() -> None:
    """Build native inputs via CPU /tokenize after DS is loaded and idle.

    Requires the exact DS checkpoint path; compares its config with the
    loaded checkpoint. Writes a new directory only. No inference or cache writes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kind", choices=("flash", "pro"), required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--public-reconstruction", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--container", default="dsv4-csa-cp8-off")
    parser.add_argument("--max-total", type=int, default=524288)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = json.loads(args.selection.read_text(encoding="utf-8"))
    dataset = (
        selected.get("dataset", "internal")
        if args.public_reconstruction
        else "internal"
    )
    if args.public_reconstruction and (
        selected.get("replay_kind") != "authorized_trajectory_reconstruction"
        or dataset not in ("terminalbench_2.1", "swe_bench_pro")
    ):
        raise ValueError(
            "Public input must be explicitly labeled authorized reconstruction"
        )
    choices = selected["requests"]
    ids = [row["source_id"] for row in choices]
    if len(ids) != 6 or len(set(ids)) != 6:
        raise ValueError("Exactly6 unique original IDs required")
    container_info = json.loads(
        subprocess.check_output(["docker", "inspect", args.container])
    )[0]
    mount = validate_checkpoint_mount(container_info, args.checkpoint, args.model_kind)
    verified_assets, unavailable = {}, []
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        actual = subprocess.run(
            ["docker", "exec", args.container, "cat", "/pro_model/" + name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        original = args.checkpoint / name
        if actual.returncode or not original.is_file():
            unavailable.append(name)
            continue
        if original.read_bytes() != actual.stdout:
            raise ValueError("Readable checkpoint/tokenizer bytes differ")
        verified_assets[name] = hashlib.sha256(actual.stdout).hexdigest()
        if (
            name == "config.json"
            and json.loads(actual.stdout).get("model_type") == "glm_moe_dsa"
        ):
            raise ValueError("GLM remains loaded; do not run DS builder yet")
    identity = dict(
        model_kind=args.model_kind,
        model_display_name={"flash": "V4-Flash", "pro": "V4-Pro"}[args.model_kind],
        model_config_sha256=verified_assets.get("config.json"),
        checkpoint_mount_identity=dict(
            container_id=container_info["Id"],
            started_at=container_info["State"]["StartedAt"],
            mount=mount,
            unavailable_file_hashes=unavailable,
            basis="running_readonly_model_bind_and_live_native_tokenization",
        ),
    )
    with urllib.request.urlopen(args.endpoint + "/v1/models", timeout=15) as response:
        models = json.load(response)["data"]
    if len(models) != 1:
        raise ValueError("Ambiguous served model")
    model = models[0]["id"]
    assets = {k: v for k, v in verified_assets.items() if k != "config.json"}

    def encode(body: dict) -> list[int]:
        """Use the loaded server's native template without trimming source content."""
        return native_tokens(args.endpoint, model, preserve_ds_history(body))

    packed = array("i")
    chains, audit = [], []
    aliases = {}
    for index, choice in enumerate(choices):
        if args.public_reconstruction:
            location = dict(source_file=choice["reconstructed_file"], source_line=1)
            source = json.loads(
                Path(location["source_file"]).read_text(encoding="utf-8")
            )
            tokens, history, count = encode_reconstructed(
                source, choice, encode, args.max_total
            )
            identity_id = safe_source_id(dataset, choice["source_id"])
            aliases[identity_id] = choice["source_id"]
            source_history, source_append = choice["source_H"], choice["source_A"]
            content_hash = canonical_hash(source["request_body"])
        else:
            location = choice.get("original", choice)
            with Path(location["source_file"]).open() as stream:
                source = next(
                    json.loads(line)
                    for n, line in enumerate(stream, 1)
                    if n == location["source_line"]
                )
            tokens, history, count = encode_original(
                source, choice, encode, args.max_total
            )
            identity_id = choice["source_id"]
            source_history = source["cache_read_tokens"]
            source_append = source["prompt_tokens"] - source_history
            content_hash = body_hash(source["request_body"])
        seed_offset = len(packed)
        packed.extend(tokens[:history])
        offset = len(packed)
        packed.extend(tokens)
        row = dict(
            round=1,
            source_request_id=identity_id,
            source_file=location["source_file"],
            source_line=location["source_line"],
            body_sha256=content_hash,
            prompt_tokens=len(tokens),
            token_offset=offset,
            reusable_input_prefix_tokens=history,
            new_input_tokens=len(tokens) - history,
            output_budget=choice["O"],
            cluster=choice["cluster"],
            source_history=source_history,
            source_append=source_append,
            history_source="controlled_native_previous_prompt_LCP",
            input_mode="controlled_native_content_prefix",
            prefix_message_count=count,
            expected_storage_aligned_history=history // 256 * 256,
            expected_storage_aligned_append=len(tokens) - history // 256 * 256,
            input_token_sha256=hashlib.sha256(array("i", tokens).tobytes()).hexdigest(),
        )
        if args.public_reconstruction:
            row.update(
                original_source_id=choice["source_id"],
                replay_kind="authorized_trajectory_reconstruction",
                input_mode="reconstructed_native_content_prefix",
            )
        chains.append(
            dict(
                chain=index,
                first_timed_index=1,
                rounds=[
                    dict(
                        round=0,
                        source_request_id="__native_prefix__:" + identity_id,
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
        audit.append(dict(case=index, **row))
        print(
            json.dumps(
                dict(case=index, H=history, A=len(tokens) - history, O=choice["O"])
            ),
            flush=True,
        )
    final_info = json.loads(
        subprocess.check_output(["docker", "inspect", args.container])
    )[0]
    validate_checkpoint_mount(final_info, args.checkpoint, args.model_kind)
    if final_info["Id"] != container_info["Id"]:
        raise ValueError("Native tokenizer container changed during encoding")
    manifest = dict(
        schema="ds_native_content_prefix_v1042",
        model=model,
        **identity,
        dataset=dataset,
        selected_count=6,
        measured_request_count=6,
        selected_source_ids=list(aliases) if args.public_reconstruction else ids,
        chains=chains,
        unavailable=[],
        replacements=[],
        replacement_count=0,
        continuation_of=None,
        native_assets=dict(
            source=str(args.checkpoint),
            hashes=assets,
            method="original_content_server_native_tokenize",
        ),
        source_cohort="targeted_internal6_native_content_prefix",
        historical_reasoning_retained=True,
        native_render_policy=dict(
            drop_thinking=False,
            reasoning_content_alias="reasoning",
            enable_thinking_when_source_reasoning=True,
        ),
        native_content_warmup=selected.get("native_content_warmup", False),
        selection_sha256=hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        performance_metrics_used_for_selection=selected.get(
            "performance_metrics_used_for_selection", False
        ),
        provider_cache_boundary_reconstructed=False,
        token_file_sha256=hashlib.sha256(packed.tobytes()).hexdigest(),
    )
    if args.public_reconstruction:
        manifest.update(
            source_id_alias_map=aliases,
            replay_kind="authorized_trajectory_reconstruction",
            source_cohort="authorized_public15_native_content_prefix",
            exact_original_http_request=False,
        )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "tokens.i32le").open("xb") as stream:
        packed.tofile(stream)
    for name, obj in (
        ("manifest.json", manifest),
        ("audit.json", audit),
        ("COMPLETE.json", dict(inputs_ready=True, requests=6, inference_requests=0)),
    ):
        (args.output / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
