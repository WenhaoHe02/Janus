# SPDX-License-Identifier: Apache-2.0
"""Prepare verified multi-round inputs using the serving model's native tokenizer."""

# Standard
import argparse
from array import array
from collections.abc import Callable
import json
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request


def common_prefix(left: list[int], right: list[int]) -> int:
    """Count identical leading tokens without rounding to cache block boundaries."""
    for offset, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return offset
    return min(len(left), len(right))


def native_tokens(endpoint: str, model: str, body: dict[str, Any], *, chat_template: str | None = None) -> list[int]:
    """Tokenize original messages/tools with the actual server chat template.

    Args:
        endpoint: Local server URL; only its CPU /tokenize route is called.
        model: Served model name.
        body: Original OpenAI-format request body, not request headers.
        chat_template: Optional verbatim template from the served model assets.
    Returns:
        Complete native model token IDs; no cropping, repetition or synthetic text.
    Raises:
        ValueError: Missing or invalid returned tokens.
        HTTPError: The server cannot represent the original request.
    """
    payload = dict(
        model=model,
        messages=body["messages"],
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    for key in ("tools", "chat_template_kwargs"):
        if body.get(key) is not None:
            payload[key] = body[key]
    if chat_template is not None:
        payload["chat_template"] = chat_template
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/tokenize",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    tokens = result.get("tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or token < 0 for token in tokens)
        or result.get("count") != len(tokens)
    ):
        raise ValueError("invalid native tokenizer response")
    return tokens


def prepare(
    source: Path, output: Path, endpoint: str, model: str, max_total: int,
    min_reusable_prefix: int = 65536,
    tokenize_body: Callable[[dict[str, Any]], list[int]] | None = None,
) -> None:
    """Save all compatible observed chains and explicitly report exclusions.

    Args:
        source: Response-verified chain manifest with original file/line pointers.
        output: New output directory; existing evidence is never overwritten.
        endpoint: Live local tokenizer endpoint; run between, not during, timings.
        model: Actual served model alias.
        max_total: Maximum native prompt plus source output budget.
        min_reusable_prefix: Minimum continuation input LCP. Zero preserves short
            early rounds of a genuine long chain; default retains the old filter.
        tokenize_body: Optional verified original-template CPU tokenizer.
    Raises:
        FileExistsError: Output already exists.
        ValueError: Original identity changed or no complete chain is usable.

    Notes:
        Chains are trace-driven replay segments, not generated closed-loop agents.
        Each next input retains the source assistant reply, not our sampled reply.
        The first round is a cold bootstrap, not a separately seeded cache hit.
    """
    data = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=False)
    packed = array("i")
    accepted, excluded = [], []
    for chain_index, chain in enumerate(data["unbranched_chains"]):
        rounds = []
        previous: list[int] = []
        staged: list[list[int]] = []
        reason = None
        for ordinal, identity in enumerate(chain):
            provenance = data["requests"][identity]
            with Path(provenance["source_file"]).open() as stream:
                original = json.loads(
                    next(
                        line
                        for number, line in enumerate(stream, 1)
                        if number == provenance["source_line"]
                    )
                )
            if original["request_id"] != identity:
                raise ValueError("source request changed")
            try:
                tokens = (tokenize_body(original["request_body"]) if tokenize_body
                          else native_tokens(endpoint, model, original["request_body"]))
            except urllib.error.HTTPError as error:
                reason = f"native tokenizer HTTP {error.code} at round {ordinal}"
                break
            history = common_prefix(previous, tokens) if previous else 0
            budget = int(original["completion_tokens"])
            if len(tokens) + budget > max_total or budget <= 0:
                reason = f"round {ordinal} exceeds native capacity or lacks output"
                break
            if previous and history < min_reusable_prefix:
                reason = f"round {ordinal} has less than {min_reusable_prefix} reusable native tokens"
                break
            rounds.append(
                dict(
                    round=ordinal,
                    source_request_id=identity,
                    source_file=provenance["source_file"],
                    source_line=provenance["source_line"],
                    prompt_tokens=len(tokens),
                    reusable_input_prefix_tokens=history,
                    new_input_tokens=len(tokens) - history,
                    output_budget=budget,
                    provider_reported_history=provenance["reported_history"],
                    provider_reported_prompt=provenance["reported_prompt"],
                )
            )
            staged.append(tokens)
            previous = tokens
        if reason:
            excluded.append(dict(chain=chain_index, reason=reason, request_ids=chain))
            continue
        for row, tokens in zip(rounds, staged, strict=True):
            row["token_offset"] = len(packed)
            packed.extend(tokens)
        accepted.append(dict(chain=chain_index, rounds=rounds))
    if sys.byteorder != "little":
        packed.byteswap()
    with (output / "tokens.i32le").open("wb") as stream:
        packed.tofile(stream)
    manifest = dict(
        model=model,
        native_tokenizer_endpoint=endpoint + "/tokenize",
        max_total=max_total,
        min_reusable_prefix=min_reusable_prefix,
        source=str(source),
        chains=accepted,
        excluded=excluded,
        GPU_inference_requests=0,
        measurements_complete=False,
        source_provider_session_id_available=False,
        replay=(
            "original messages and original prior replies; trace-driven not closed-loop"
        ),
        startup="cold first round; preserve SSD state across subsequent rounds",
        output="source completion token count used as target-model output budget",
        history="native input-token LCP; actual restored token count must be measured",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            dict(
                accepted_chains=len(accepted),
                excluded_chains=len(excluded),
                rounds=sum(len(c["rounds"]) for c in accepted),
                GPU_inference_requests=0,
            )
        ),
        flush=True,
    )
    if not accepted:
        raise ValueError("no complete native-tokenized replay chain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--max-total", type=int, default=278528)
    parser.add_argument("--min-reusable-prefix", type=int, default=65536)
    args = parser.parse_args()
    prepare(args.source, args.output, args.endpoint, args.model, args.max_total,
            args.min_reusable_prefix)
