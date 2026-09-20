# SPDX-License-Identifier: Apache-2.0
"""Create a hash-checked source-only archive for an isolated build check."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def bundle(source: Path, output: Path) -> int:
    """Archive only package build sources already listed in the review manifest.

    Returns file count. Fails on modified source, an existing archive, or an
    escaped path. Does not include models, experiments, results or bytecode.
    """
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    roots = {"lmcache", "csrc", "requirements"}
    standalone = {"setup.py", "pyproject.toml", "MANIFEST.in", "LICENSE", "README.md"}
    count = 0
    with tarfile.open(output, "x:gz") as archive:
        for row in manifest["files"]:
            path = Path(row["path"])
            if path.parts[0] not in roots and row["path"] not in standalone:
                continue
            full = source / path
            if path.is_absolute() or ".." in path.parts or full.is_symlink():
                raise ValueError("Unsafe build-source path")
            if hashlib.sha256(full.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Build source differs from audited manifest")
            archive.add(full, arcname=path.as_posix(), recursive=False)
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(bundle(args.source, args.output))
