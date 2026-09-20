# SPDX-License-Identifier: Apache-2.0
"""Audit content-defined native prefixes without transferring provider token counts."""

# Standard
import argparse
import json
from pathlib import Path
import subprocess

# Local
from build_existing29_model_inputs_v957 import body_hash
from prepare_native_chain_replay_v587 import common_prefix


def prefix_before_incoming_turn(
    body: dict, *, include_response: bool = True
) -> tuple[dict, int]:
    """Return the original message prefix ending at the last assistant response.

    This defines a controlled history snapshot, not a reconstruction of the
    provider's cache-hit boundary. The full timed input is unchanged. Raise
    ValueError if no assistant prefix or no following incoming messages exist.
    """
    messages = body["messages"]
    last = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
    )
    if last < 0 or last == len(messages) - 1:
        raise ValueError("No completed assistant response followed by an incoming turn")
    if any(m["role"] not in ("user", "tool", "system") for m in messages[last + 1 :]):
        raise ValueError("Unsupported incoming message roles")
    count = last + 1 if include_response else last
    if count == 0:
        raise ValueError("No preceding prompt messages")
    prefix = dict(body, messages=messages[:count])
    return prefix, count


def main() -> None:
    """Audit long candidates and cases0/2/8 under both template policies on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/glm_native_boundary_v1024")
    )
    parser.add_argument("--exclude-last-response", action="store_true")
    args = parser.parse_args()
    source = Path("/tmp/glm_trace_reasoning_v1023/RESULTS.json")
    rows = [
        r
        for r in json.loads(source.read_text())
        if r.get("long_candidate_cluster") or r["frozen_case"] in (0, 2, 8)
    ]
    root = args.output
    root.mkdir(exist_ok=False)
    results = []
    with subprocess.Popen(
        [
            "docker",
            "exec",
            "-i",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
            "dsv4-csa-cp8-off",
            "python3",
            "/tmp/glm_native_tokenizer_v969.py",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as worker:
        if worker.stdin is None or worker.stdout is None:
            raise RuntimeError("Missing CPU tokenizer pipe")
        ready = json.loads(worker.stdout.readline())
        if not ready.get("ready"):
            raise RuntimeError("CPU tokenizer failed to initialize")
        (root / "assets.json").write_text(json.dumps(ready, indent=2))
        (root / "audit_source.py").write_bytes(Path(__file__).read_bytes())

        def encode(body: dict) -> list[int]:
            """Encode original content using the existing checkpoint CPU worker."""
            worker.stdin.write(json.dumps(body) + "\n")
            worker.stdin.flush()
            response = json.loads(worker.stdout.readline())
            if "tokens" not in response:
                raise RuntimeError(f"CPU encoder failed: {response}")
            return response["tokens"]

        try:
            for row in rows:
                with Path(row["source_file"]).open() as stream:
                    record = next(
                        json.loads(line)
                        for n, line in enumerate(stream, 1)
                        if n == row["source_line"]
                    )
                body = record["request_body"]
                if body_hash(body) != row["body_sha256"]:
                    raise RuntimeError("Original request body changed")
                prefix, count = prefix_before_incoming_turn(
                    body, include_response=not args.exclude_last_response
                )
                item = {
                    k: row.get(k)
                    for k in (
                        "source_request_id",
                        "frozen_case",
                        "source_history",
                        "source_append",
                        "output_budget",
                        "long_candidate_cluster",
                        "body_sha256",
                    )
                }
                item.update(
                    history_messages=count,
                    incoming_messages=len(body["messages"]) - count,
                    provider_cache_boundary_reconstructed=False,
                    policy=(
                        "native_LCP_with_previous_prompt_prefix"
                        if args.exclude_last_response
                        else "native_LCP_with_original_prefix_through_last_assistant"
                    ),
                )
                for variant in ("default", "keep_reasoning"):
                    full_body = json.loads(json.dumps(body))
                    prefix_body = json.loads(json.dumps(prefix))
                    if variant == "keep_reasoning":
                        for request in (full_body, prefix_body):
                            request.setdefault("chat_template_kwargs", {})[
                                "clear_thinking"
                            ] = False
                    full_tokens, previous_tokens = (
                        encode(full_body),
                        encode(prefix_body),
                    )
                    history = common_prefix(full_tokens, previous_tokens)
                    item[variant] = dict(
                        prompt_tokens=len(full_tokens),
                        history=history,
                        append=len(full_tokens) - history,
                        prefix_rendered_tokens=len(previous_tokens),
                    )
                results.append(item)
                (root / "RESULTS.json").write_text(json.dumps(results, indent=2))
                print(json.dumps(item), flush=True)
        finally:
            worker.stdin.close()
            worker.wait(timeout=30)
    if worker.returncode:
        raise RuntimeError("CPU tokenizer exited unsuccessfully")
    (root / "COMPLETE.json").write_text(json.dumps(results, indent=2))
    print("NATIVE_BOUNDARY_AUDIT_COMPLETE_NO_INFERENCE", flush=True)


if __name__ == "__main__":
    main()
