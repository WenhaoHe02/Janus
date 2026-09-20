# SPDX-License-Identifier: Apache-2.0
"""Preserve the frozen internal 29 requests while preparing native model inputs."""

# Standard
import argparse
from array import array
from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import urllib.request

# Local
from prepare_native_chain_replay_v587 import common_prefix
from prepare_selected_internal_v928 import read_original


def body_hash(body: dict) -> str:
    """Return the same canonical request-body digest used by the original runner."""
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_reference_cohort(reference: dict, selection: dict, selection_hash: str) -> dict:
    """Restrict a frozen cohort without inheriting another run's completion state.

    Args:
        reference: Original native manifest, retained unchanged.
        selection: Ordered source rows selected from this manifest.
        selection_hash: SHA-256 of the selected-source document.
    Returns:
        Subset manifest with consistent counts and ancestral provenance.
    Raises:
        ValueError: Selected identities, order, count or original H/A/O differ.
    """
    rows = selection["requests"]
    selected_ids = [row["source_id"] for row in rows]
    chains = [chain for chain in reference["chains"] if
              chain["rounds"][chain["first_timed_index"]]["source_request_id"] in selected_ids]
    actual = [chain["rounds"][chain["first_timed_index"]] for chain in chains]
    if (len(set(selected_ids)) != len(selected_ids) or
            len(rows) != selection["selected_count"] or
            [row["source_request_id"] for row in actual] != selected_ids):
        raise ValueError("Subset identities, count or source order differ")
    for row, chosen in zip(actual, rows, strict=True):
        if (row["source_history"], row["source_append"], row["output_budget"]) != (
                chosen["H"], chosen["A"], chosen["O"]):
            raise ValueError("Subset changed original H/A/O")
    replacements = [row for row in reference.get("replacements", [])
                    if row["new_id"] in selected_ids]
    inherited = {key: reference.get(key) for key in (
        "source_cohort", "selected_count", "selection_path", "selection_sha256",
        "continuation_of", "completed_ids", "replacement_count", "replacements",
        "component_provenance",
    )}
    return dict(reference, chains=chains, selected_count=len(chains),
                measured_request_count=len(chains), selected_source_ids=selected_ids,
                unavailable=[], continuation_of=None, completed_ids=[],
                source_cohort=f"subset{len(chains)}_of_existing_internal29",
                replacements=replacements, replacement_count=len(replacements),
                reference_cohort_provenance=inherited, component_provenance={},
                selection_sha256=selection_hash, serving_subset_sha256=selection_hash)


def verified_model_identity(model_kind: str, expected_config: bytes, loaded_config: bytes) -> dict:
    """Identify the checkpoint by config bytes, not a possibly shared API alias.

    Args:
        model_kind: Requested experiment model, glm or flash.
        expected_config: Config from the exact checkpoint used by its launcher.
        loaded_config: Config from the container's actual /pro_model mount.
    Returns:
        Stable checkpoint hash and an unambiguous display name.
    Raises:
        ValueError: Configs differ or the model kind is unsupported.
    """
    names = {"glm": "GLM-5.2", "flash": "V4-Flash"}
    if model_kind not in names or expected_config != loaded_config:
        raise ValueError("Loaded checkpoint differs from the requested model")
    return dict(model_kind=model_kind, model_display_name=names[model_kind],
                model_config_sha256=hashlib.sha256(loaded_config).hexdigest())


def reencode_chain(
    chain: dict,
    tokenize: Callable[[dict], list[int]],
    read: Callable[[dict, str], dict],
    packed: array,
    max_total: int,
) -> dict:
    """Encode one unchanged request, preserving its existing prefix provenance.

    Args:
        chain: One frozen reference chain containing setup and timed rows.
        tokenize: Native request-body tokenizer for the new model.
        read: Exact-source loader, validated by request identity.
        packed: Destination int32 token stream, appended in place.
        max_total: Native prompt plus original output capacity limit.
    Returns:
        Same source identities and output budgets, with native token offsets.
    Raises:
        ValueError: Content, budget, prefix provenance, or capacity is invalid.
    """
    original_row = chain["rounds"][chain["first_timed_index"]]
    identity = original_row["source_request_id"]
    source = read(original_row, identity)
    body = source["request_body"]
    if body_hash(body) != original_row["body_sha256"]:
        raise ValueError(f"Frozen request content changed: {identity}")
    if source["completion_tokens"] != original_row["output_budget"]:
        raise ValueError(f"Frozen output budget changed: {identity}")
    tokens = tokenize(body)
    if not tokens or len(tokens) + original_row["output_budget"] > max_total:
        raise ValueError(f"Native request exceeds capacity: {identity}")
    mode = original_row["history_source"]
    parent: list[int] = []
    history = 0
    if mode == "verified_predecessor":
        parent_id = chain["rounds"][0]["source_request_id"]
        location = dict(source_file=str(Path(original_row["source_file"]).parent / f"{parent_id}.jsonl"), source_line=1)
        parent_body = read(location, parent_id)["request_body"]
        parent = tokenize(parent_body)
        history = common_prefix(parent, tokens)
    elif mode == "reported_source_history":
        history = min(int(original_row["source_history"]), len(tokens))
        parent = tokens[:history]
    elif mode != "none":
        raise ValueError(f"Unknown history provenance: {mode}")
    if parent and len(parent) + 4 > max_total:
        raise ValueError(f"Native predecessor exceeds capacity: {identity}")
    rows = []
    if parent:
        rows.append(dict(chain["rounds"][0], token_offset=len(packed), prompt_tokens=len(parent)))
        packed.extend(parent)
    row = dict(original_row, round=len(rows), token_offset=len(packed),
               prompt_tokens=len(tokens), reusable_input_prefix_tokens=history,
               new_input_tokens=len(tokens) - history,
               input_mode=("controlled_reported_history_seed" if mode == "reported_source_history"
                           else "verified_prefix_reuse" if parent else "first_request"))
    rows.append(row)
    packed.extend(tokens)
    return dict(chain, rounds=rows, first_timed_index=len(rows) - 1)


def main() -> None:
    """Build inputs only between formal batches and refuse cohort substitution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("glm", "flash"), required=True)
    parser.add_argument("--max-total", type=int, default=524288)
    parser.add_argument("--selection", type=Path, help="Frozen subset selected from the original 29")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = json.loads((args.reference / "manifest.json").read_text())
    if reference["selected_count"] != 29 or len(reference["chains"]) != 29:
        raise ValueError("The reference must contain the existing complete 29")
    if args.selection:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection["source_sha256"] != hashlib.sha256((args.reference / "manifest.json").read_bytes()).hexdigest():
            raise ValueError("Subset was selected from a different reference manifest")
        reference = select_reference_cohort(reference, selection,
            hashlib.sha256(args.selection.read_bytes()).hexdigest())
        reference["selection_path"] = str(args.selection.resolve())
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=15) as response:
        models = json.load(response)["data"]
    if len(models) != 1:
        raise ValueError("Ambiguous served model")
    model = models[0]["id"]
    checkpoint = (Path("/local/GLM-5.2-FP8") if args.model_kind == "glm" else
                  Path(reference["native_assets"]["flash"]["source"]))
    identity = verified_model_identity(args.model_kind, (checkpoint / "config.json").read_bytes(),
        subprocess.check_output(["docker", "exec", "dsv4-csa-cp8-off", "cat", "/pro_model/config.json"]))
    packed = array("i")
    if args.model_kind == "glm":
        template = subprocess.check_output([
            "docker", "exec", "dsv4-csa-cp8-off", "cat", "/pro_model/chat_template.jinja"
        ]).decode()

        chains = []
        with subprocess.Popen([
            "docker", "exec", "-i", "-e", "CUDA_VISIBLE_DEVICES=", "dsv4-csa-cp8-off",
            "python3", "/tmp/glm_native_tokenizer_v969.py",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8") as worker:
            assert worker.stdin is not None and worker.stdout is not None
            evidence = json.loads(worker.stdout.readline())
            if not evidence.get("ready"):
                raise ValueError("Checkpoint-native GLM encoder failed to initialize")

            def tokenize(body: dict) -> list[int]:
                """Encode unchanged original messages/tools with native GLM assets."""
                worker.stdin.write(json.dumps(body, ensure_ascii=False) + "\n")
                worker.stdin.flush()
                response = json.loads(worker.stdout.readline())
                if "tokens" not in response:
                    raise ValueError(f"Native encoder failed: {response}")
                return response["tokens"]

            try:
                for chain in reference["chains"]:
                    updated = reencode_chain(chain, tokenize, read_original, packed, args.max_total)
                    chains.append(updated)
                    row = updated["rounds"][-1]
                    print(json.dumps({key: row[key] for key in (
                        "source_request_id", "prompt_tokens", "reusable_input_prefix_tokens", "output_budget"
                    )}), flush=True)
            finally:
                worker.stdin.close()
                worker.wait(timeout=30)
        assets: dict[str, Any] = dict(checkpoint_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
                                      hashes=evidence["assets"], method=evidence["method"])
    else:
        expected = reference["native_assets"]["flash"]["hashes"]
        actual = {}
        for filename in expected:
            output = subprocess.check_output([
                "docker", "exec", "dsv4-csa-cp8-off", "sha256sum", f"/pro_model/{filename}"
            ]).decode()
            actual[filename] = output.split()[0]
        if actual != expected or expected != reference["native_assets"]["pro"]["hashes"]:
            raise ValueError("Flash and reference DS native assets do not match")
        chains = reference["chains"]
        assets = dict(reference["native_assets"], current_flash_hashes=actual,
                      method="identical_frozen_DS_tokenizer_assets_reference_tokens_reused")
    ids = [c["rounds"][c["first_timed_index"]]["source_request_id"] for c in chains]
    if ids != reference["selected_source_ids"]:
        raise ValueError("Source order or identities changed")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.model_kind == "glm":
        with (args.output / "tokens.i32le").open("wb") as stream:
            packed.tofile(stream)
    else:
        shutil.copyfile(args.reference / "tokens.i32le", args.output / "tokens.i32le")
    manifest = dict(reference, model=model, chains=chains, native_assets=assets,
                    **identity,
                    continuation_of=None, completed_ids=[],
                    reference_continuation_of=reference.get("continuation_of"),
                    reference_manifest_sha256=hashlib.sha256((args.reference / "manifest.json").read_bytes()).hexdigest(),
                    token_file_sha256=hashlib.sha256((args.output / "tokens.i32le").read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(dict(ready=True, selected=len(chains), model=model, output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
