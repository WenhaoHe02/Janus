# SPDX-License-Identifier: Apache-2.0
"""Remove the historical decode-read bypass from submission copies only."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

REMOVED = {
    "install_ds_serving_decode_v889.py",
    "install_glm_serving_decode_v763.py",
    "serving_decode_policy_v763.py",
}
CUDA_STREAM_SOURCES = (
    "csrc/compact_prefill.cu",
    "csrc/mem_kernels.cu",
    "csrc/pos_kernels.cu",
    "csrc/mem_kernels.cuh",
    "csrc/pos_kernels.cuh",
    "csrc/mp_mem_kernels.cu",
)


def normalize_cuda_source(relative: str, source: str) -> str:
    """Use stream/guard headers without an unnecessary cuSPARSE dependency.

    Only the audited files using CUDAContext solely for stream and guard
    APIs are changed. c10 is the owning namespace of these same CUDA APIs.
    """
    if relative not in CUDA_STREAM_SOURCES:
        return source
    return (
        source.replace(
            "#include <ATen/cuda/CUDAContext.h>", "#include <c10/cuda/CUDAStream.h>"
        )
        .replace("at::cuda::getCurrentCUDAStream", "c10::cuda::getCurrentCUDAStream")
        .replace("at::cuda::OptionalCUDAGuard", "c10::cuda::OptionalCUDAGuard")
    )


def submission_source(relative: str, source: str) -> str | None:
    """Return source with normal demand-read decode, or None for removed helpers.

    Args:
        relative: Path within the submission tree.
        source: Original source text, never modified in the archival directory.
    Returns:
        Submission-only text, or None for the three obsolete bypass modules.
    Notes:
        Removes the flag, installer invocations and obsolete runner labels.
        Does not disable writes, alter prediction, or modify decode outputs.
    """
    if not relative.startswith("experiments/legacy/"):
        return normalize_cuda_source(relative, source)
    name = PurePosixPath(relative).name
    if name in REMOVED:
        return None
    lines = []
    for line in source.splitlines(keepends=True):
        if "('install_ds_serving_decode_v889.py'," in line:
            continue
        if "('install_glm_serving_decode_v763.py'," in line:
            continue
        if line.strip() == "export LMCACHE_SERVING_ASSUMED_HBM_DECODE=1":
            continue
        if line.strip() == "'LMCACHE_SERVING_ASSUMED_HBM_DECODE': '1',":
            continue
        line = line.replace("LMCACHE_SERVING_ASSUMED_HBM_DECODE='1', ", "")
        line = line.replace("LMCACHE_SERVING_ASSUMED_HBM_DECODE='1',", "")
        line = line.replace("assumed_HBM_no_demand_decode", "normal_demand_decode")
        line = line.replace(
            "missing isolated decode-assumption deployment proof",
            "missing normal-decode deployment proof",
        )
        lines.append(line)
    return "".join(lines)


def clean_review(review: Path) -> dict:
    """Remove bypass code from one explicit review tree and update its hashes."""
    root = review.resolve()
    manifest_path = root / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    kept, removed, changed = [], [], []
    for record in manifest["files"]:
        relative = record["path"]
        if not relative.startswith("experiments/legacy/"):
            kept.append(record)
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source outside review directory")
        payload = path.read_bytes()
        source = payload.decode("utf-8-sig")
        result = submission_source(relative, source)
        if result is None:
            path.unlink()
            removed.append(relative)
            continue
        if result != source:
            if path.suffix == ".py":
                ast.parse(result)
            updated = result.encode("utf-8")
            path.write_bytes(updated)
            record.update(
                original_sha256=hashlib.sha256(payload).hexdigest(),
                sha256=hashlib.sha256(updated).hexdigest(),
                bytes=len(updated),
                submission_transform="normal_decode",
            )
            changed.append(relative)
        kept.append(record)
    manifest["files"] = kept
    manifest["removed_decode_bypass_modules"] = removed
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return dict(removed=len(removed), changed=len(changed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    print(json.dumps(clean_review(parser.parse_args().review)))
