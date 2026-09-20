# SPDX-License-Identifier: Apache-2.0
"""Apply one model-family overlay to a matching fresh installation."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path


def apply_overlay(
    source: Path,
    lmcache_root: Path,
    vllm_root: Path,
    model: str,
    check_only: bool = False,
) -> dict:
    """Validate all target hashes, then apply the selected source overlay.

    Args:
        source: Submission repository root containing overlays/.
        lmcache_root: Isolated installation of the submission's default LMCache.
        vllm_root: Isolated installation of the pinned baseline vLLM.
        model: flash, pro or glm.
        check_only: Check compatibility without writing files.
    Returns:
        Applied/current counts and family. Raises ValueError before writing if
        a target differs from both the pinned base and selected overlay.
    Notes:
        Switch model families using a fresh environment; never hot-patch a live
        worker or silently accept an unknown local modification.
    """
    if model not in {"flash", "pro", "glm"}:
        raise ValueError("Unknown model family")
    family = "glm" if model == "glm" else "dsv4"
    root = source / "overlays" / family
    manifest = json.loads((root / "manifest.json").read_text())
    pending, already = [], 0
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid overlay path")
        if relative.parts[0] not in {"lmcache", "vllm"}:
            raise ValueError("Overlay target outside supported packages")
        package = lmcache_root if relative.parts[0] == "lmcache" else vllm_root
        target = package.joinpath(*relative.parts[1:])
        if target.is_symlink() or not target.resolve().is_relative_to(
            package.resolve()
        ):
            raise ValueError("Overlay target escapes package")
        payload = (root / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["after_sha256"]:
            raise ValueError(f"Overlay source changed: {relative}")
        current = (
            hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        )
        if current == row["after_sha256"]:
            already += 1
        elif current == row["before_sha256"]:
            pending.append((target, payload))
        else:
            raise ValueError(f"Incompatible installation: {relative}")
    if not check_only:
        for target, payload in pending:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".janus-pending")
            with temporary.open("xb") as stream:
                stream.write(payload)
            temporary.replace(target)
    return dict(
        model=model,
        family=family,
        changed=len(pending),
        current=already,
        check_only=check_only,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--lmcache-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--model", choices=("flash", "pro", "glm"), required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            apply_overlay(
                args.source,
                args.lmcache_root,
                args.vllm_root,
                args.model,
                args.check_only,
            )
        )
    )
