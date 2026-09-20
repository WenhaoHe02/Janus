# SPDX-License-Identifier: Apache-2.0
"""Compile actual native seed/hit paths before admitting external requests."""

# Standard
import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import sys

# Local
from startup_io import StartupClient


def load_inputs(inputs: Path, max_model_len: int) -> tuple[dict, array, list[dict]]:
    """Read a private native manifest, preserving exact H/A and token IDs.

    Returns manifest, int32 token array and unique selected request rows.
    Rejects duplicate IDs, invalid slices, negative IDs and shapes exceeding
    the configured context limit including the four warmup output tokens.
    """
    manifest = json.loads((inputs / "manifest.json").read_text())
    tokens = array("i")
    path = inputs / "tokens.i32le"
    if tokens.itemsize != 4 or path.stat().st_size % 4:
        raise ValueError("Native inputs must be little-endian int32")
    with path.open("rb") as stream:
        tokens.fromfile(stream, path.stat().st_size // 4)
    if sys.byteorder != "little":
        tokens.byteswap()
    rows, seen = [], set()
    for chain in manifest["chains"]:
        index = chain["first_timed_index"]
        if index not in (0, 1) or len(chain["rounds"]) != index + 1:
            raise ValueError("Unexpected selected-session layout")
        row = chain["rounds"][index]
        identity = row["source_request_id"]
        start, count, history = (
            row["token_offset"],
            row["prompt_tokens"],
            row["reusable_input_prefix_tokens"],
        )
        if (
            not isinstance(identity, str)
            or not identity
            or identity in seen
            or any(type(value) is not int for value in (start, count, history))
            or start < 0
            or count <= 0
            or not 0 <= history < count
            or start + count > len(tokens)
            or count + 4 > max_model_len
            or any(value < 0 for value in tokens[start : start + count])
        ):
            raise ValueError(
                "Invalid native request identity, bounds or context length"
            )
        seen.add(identity)
        rows.append(row)
    if not rows or len(rows) != manifest["selected_count"]:
        raise ValueError("Warmup cohort is empty or incomplete")
    return manifest, tokens, rows


def warm_cases(
    client: StartupClient,
    tokens: array,
    rows: list[dict],
    output: Path,
    artifact_id: str,
) -> list[dict]:
    """Warm each original prefix and append through public APIs, then clear its cache.

    The caller must own a dedicated pre-admission cache. No formal requests or
    output budgets are changed, no worker script is uploaded, and no legacy
    launcher is imported. Publication and read-hit evidence are mandatory.
    """
    output.mkdir(parents=True, exist_ok=False)
    events = []
    with (output / "preparation.jsonl").open("x", encoding="utf-8") as log:
        client.clear_dedicated_cache()
        for index, row in enumerate(rows):
            start = row["token_offset"]
            prompt = list(tokens[start : start + row["prompt_tokens"]])
            history = row["reusable_input_prefix_tokens"]
            tag = f"janus-warm:{artifact_id}:{index}"
            if history:
                seed = client.complete(prompt[:history], tag)
                # Small histories have no complete cache chunk to publish.
                if history >= 256:
                    client.wait_publication(seed["request_id"], history)
            result = client.complete(prompt, tag)
            hit = client.require_hit(result["request_id"], history)
            client.wait_publication(result["request_id"], len(prompt), hit)
            event = dict(
                source_request_id=row["source_request_id"],
                phase="actual_content_warmup_excluded",
                history=history,
                append=len(prompt) - history,
                restored_tokens=hit,
                input_sha256=hashlib.sha256(array("i", prompt).tobytes()).hexdigest(),
                **result,
            )
            events.append(event)
            log.write(json.dumps(event) + "\n")
            log.flush()
            client.clear_dedicated_cache()
    return events


def main() -> None:
    """Warm a dedicated Tutti deployment from its declared native cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--rank-endpoints", nargs=8, required=True)
    parser.add_argument("--worker-env", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--dedicated-cache", action="store_true", required=True)
    args = parser.parse_args()
    manifest, tokens, rows = load_inputs(args.inputs, args.max_model_len)
    output = Path(os.environ["JANUS_START_DIRECTORY"])
    artifact_id = os.environ["JANUS_ARTIFACT_ID"]
    client = StartupClient(
        args.endpoint, args.rank_endpoints, manifest["model"], output / "server.log"
    )
    expected = json.loads(args.worker_env.read_text())
    expected["JANUS_ARTIFACT_ID"] = artifact_id
    client.verify_workers(expected)
    client.drain()
    client.verify_loaded_native(Path(os.environ["JANUS_COUNTED_BUILD_RECEIPT"]))
    events = warm_cases(client, tokens, rows, output / "real_path_warmup", artifact_id)
    client.verify_workers(expected)
    client.verify_loaded_native(Path(os.environ["JANUS_COUNTED_BUILD_RECEIPT"]))
    receipt = dict(
        status="complete",
        artifact_id=artifact_id,
        warmup_input_mode="actual_native_content",
        covered_requests=len(events),
        expected_requests=len(rows),
        manifest_sha256=hashlib.sha256(
            (args.inputs / "manifest.json").read_bytes()
        ).hexdigest(),
        shapes=sorted({(event["history"], event["append"]) for event in events}),
        rank_count=8,
        source_output_modified=False,
        dedicated_warmup_cache_cleared=True,
    )
    Path(os.environ["JANUS_WARMUP_RECEIPT"]).write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
