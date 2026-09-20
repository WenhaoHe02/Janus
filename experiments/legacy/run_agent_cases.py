# SPDX-License-Identifier: Apache-2.0
"""Run accuracy or decode-coverage cases from a frozen Agent token corpus."""

from __future__ import annotations

# Standard
import argparse
from array import array
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import urllib.request

# Local experiment helpers
from run_dsv4_nsys_formal_profile import _stream_formal
from run_glm52_recovery_smoke import stream_completion, wait_reuse_ready


def load_tokens(path: Path) -> array:
    """Read a little-endian signed-int32 corpus."""
    tokens = array("i")
    with path.open("rb") as input_file:
        tokens.fromfile(input_file, path.stat().st_size // 4)
    if tokens.itemsize != 4:
        raise RuntimeError("this platform does not provide 32-bit integers")
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens


def clear_disk_cache() -> list[dict]:
    """Clear all rank-local disk metadata and raw allocation generations."""
    statuses = []
    for rank in range(8):
        request = urllib.request.Request(
            f"http://127.0.0.1:{7000 + rank}"
            "/cache/clear?locations=LocalDiskBackend",
            method="DELETE",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            status = json.loads(response.read().decode())
        if status.get("status") != "success":
            raise RuntimeError(f"rank {rank} cache clear failed: {status}")
        statuses.append(status)
    return statuses


def slice_case(
    corpus: array,
    history_tokens: int,
    append_tokens: int,
    seed: int,
) -> tuple[list[int], list[int], list[int], dict[str, int]]:
    """Select deterministic contiguous Agent content and unrelated padding."""
    required = history_tokens + append_tokens
    if required + 256 > len(corpus):
        raise RuntimeError(
            f"case needs {required + 256} tokens but corpus has {len(corpus)}"
        )
    start_capacity = len(corpus) - required - 256
    start = (seed * 104729) % (start_capacity + 1)
    history_end = start + history_tokens
    append_end = history_end + append_tokens
    padding_start = append_end
    history = list(corpus[start:history_end])
    append = list(corpus[history_end:append_end])
    padding = list(corpus[padding_start : padding_start + 256])
    return history, append, padding, {
        "corpus_start": start,
        "history_start": start,
        "history_end": history_end,
        "append_start": history_end,
        "append_end": append_end,
        "padding_start": padding_start,
        "padding_end": padding_start + 256,
    }


def slice_explicit_case(
    corpus: array,
    history_tokens: int,
    append_tokens: int,
    corpus_start: int,
    padding_start: int,
) -> tuple[list[int], list[int], list[int], dict[str, int]]:
    """Read one exact request packed at caller-recorded corpus offsets."""
    history_end = corpus_start + history_tokens
    append_end = history_end + append_tokens
    padding_end = padding_start + 256
    if min(corpus_start, padding_start) < 0 or max(append_end, padding_end) > len(corpus):
        raise RuntimeError("explicit trace case lies outside the packed corpus")
    return (
        list(corpus[corpus_start:history_end]),
        list(corpus[history_end:append_end]),
        list(corpus[padding_start:padding_end]),
        {
            "corpus_start": corpus_start,
            "history_start": corpus_start,
            "history_end": history_end,
            "append_start": history_end,
            "append_end": append_end,
            "padding_start": padding_start,
            "padding_end": padding_end,
        },
    )


def main() -> None:
    """Execute frozen cases without any synthetic token generator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=None,
        help="Override generated tokens while preserving the trace decode metadata.",
    )
    parser.add_argument(
        "--mode", choices=("accuracy", "decode-coverage"), required=True
    )
    args = parser.parse_args()
    if args.output_tokens is not None and args.output_tokens <= 0:
        raise ValueError("--output-tokens must be positive")

    source_manifest = json.loads(args.corpus_manifest.read_text())
    if source_manifest.get("synthetic_token_generator") is not False:
        raise RuntimeError("corpus provenance is not explicitly non-synthetic")
    cases = [json.loads(line) for line in args.cases.read_text().splitlines()]
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    cases = cases[args.start_index :]
    if args.case_limit is not None:
        cases = cases[: args.case_limit]
    corpus = load_tokens(args.corpus)
    if len(corpus) != int(source_manifest["token_count"]):
        raise RuntimeError("corpus length does not match its provenance manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "mode": args.mode,
        "source_manifest": source_manifest,
        "source_manifest_path": str(args.corpus_manifest),
        "cases_path": str(args.cases),
        "start_index": args.start_index,
        "cases": cases,
        "synthetic_token_generator": False,
    }
    (args.output_dir / "input_provenance.json").write_text(
        json.dumps(suite_manifest, ensure_ascii=False, indent=2) + "\n"
    )

    for case_index, case in enumerate(cases):
        history_tokens = int(case["history"])
        append_tokens = int(case["append"])
        decode_tokens = int(case["decode"])
        seed = int(case["seed"])
        path = args.output_dir / (
            f"{args.mode}_h{history_tokens}_a{append_tokens}_d{decode_tokens}.json"
        )
        if path.exists():
            print(f"SKIP_EXISTING case={case_index} path={path}")
            continue

        if "corpus_start" in case and "padding_start" in case:
            history, append, padding, slices = slice_explicit_case(
                corpus,
                history_tokens,
                append_tokens,
                int(case["corpus_start"]),
                int(case["padding_start"]),
            )
        else:
            history, append, padding, slices = slice_case(
                corpus,
                history_tokens,
                append_tokens,
                seed,
            )
        clear_statuses = clear_disk_cache()
        # vLLM schedules these requests in 8,192-token prefill batches.  If
        # the final batch contains fewer than one 256-token cache chunk,
        # LMCache has no MemoryObj on the terminal store call and cannot
        # finalize the layer-major generation accumulated by earlier calls.
        # Padding only the cold seed forces a cacheable terminal call; the
        # formal accuracy/coverage request remains the exact Agent slice.
        final_batch_tokens = history_tokens % 8192
        cold_padding_tokens = 256 if final_batch_tokens < 256 else 0
        cold_tokens = history + padding[:cold_padding_tokens]
        cold = stream_completion(cold_tokens, skip_save=False)
        if cold.request_id is None:
            raise RuntimeError("cold request returned no request ID")
        ready = wait_reuse_ready(cold.request_id)
        time.sleep(0.2)
        output_tokens = (
            args.output_tokens
            if args.output_tokens is not None
            else (1 if args.mode == "accuracy" else decode_tokens)
        )
        formal = _stream_formal(
            history + append,
            output_tokens=output_tokens,
            nsys_command=["/bin/true"],
            session="disabled",
        )
        payload = {
            "configuration": case,
            "input_provenance": {
                "source": source_manifest["source"],
                "dataset": source_manifest["dataset"],
                "parquet_skip_rows": source_manifest["parquet_skip_rows"],
                "serialized_fields": source_manifest["serialized_fields"],
                "corpus_slices": slices,
                "synthetic_token_generator": False,
            },
            "clear_statuses": clear_statuses,
            "cold_padding_tokens": cold_padding_tokens,
            "cold": asdict(cold),
            "ready": ready,
            "formal": formal,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(
            f"CASE_DONE index={case_index} history={history_tokens} "
            f"append={append_tokens} decode={output_tokens} "
            f"request={formal['request_id']}"
        )
    print(f"SUITE_DONE outdir={args.output_dir}")


if __name__ == "__main__":
    main()
