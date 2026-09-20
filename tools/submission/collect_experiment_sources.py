# SPDX-License-Identifier: Apache-2.0
"""Archive only literal source dependencies of the selected historical scripts."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

# Local
from build_review import EXPERIMENT_ROOTS, source_dependencies


def collect(source_root: Path, archive: Path) -> dict:
    """Read source under one root and create a new source-only archive.

    Returns file hashes and unresolved basenames. Rejects symlinks and large
    source files; does not execute source or access devices/containers.
    """
    pending, seen, missing, files = list(EXPERIMENT_ROOTS), set(), [], []
    with tarfile.open(archive, "x:gz") as bundle:
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if Path(name).name != name:
                raise ValueError("Only flat source basenames are allowed")
            source = source_root / name
            if not source.is_file():
                missing.append(name)
                continue
            if source.is_symlink() or source.stat().st_size > 5 * 1024 * 1024:
                raise ValueError(f"Unsafe source: {name}")
            payload = source.read_bytes()
            pending.extend(
                source_dependencies(payload.decode("utf-8-sig"), source.suffix) - seen
            )
            bundle.add(source, arcname=name, recursive=False)
            files.append(dict(name=name, sha256=hashlib.sha256(payload).hexdigest()))
    receipt = dict(
        files=files,
        missing=sorted(missing),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    archive.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    """Collect source dependencies, with no server or inference side effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    receipt = collect(args.source_root, args.archive)
    print(json.dumps(dict(files=len(receipt["files"]), missing=receipt["missing"])))


if __name__ == "__main__":
    main()
