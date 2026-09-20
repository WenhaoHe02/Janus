# SPDX-License-Identifier: Apache-2.0
"""Refresh submission tooling only, then verify a generated review manifest."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

# Local
from release_decode_cleanup import CUDA_STREAM_SOURCES, normalize_cuda_source


def normalize_line_endings(review: Path, records: dict[str, dict]) -> int:
    """Normalize audited text to LF and refresh overlay hashes, not source logic.

    Args:
        review: Source-only review tree whose hashes were already checked.
        records: Mutable manifest entries keyed by relative path.
    Returns:
        Count of files whose CRLF byte sequences were normalized.
    Raises:
        UnicodeDecodeError: A supposedly textual source contains invalid UTF-8.
    """
    changed = 0
    for relative, row in records.items():
        path = review / relative
        raw = path.read_bytes()
        raw.decode("utf-8-sig")
        payload = raw.replace(b"\r\n", b"\n")
        if payload != raw:
            path.write_bytes(payload)
            row.setdefault("before_lf_sha256", hashlib.sha256(raw).hexdigest())
            row.update(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
            changed += 1
    for model in ("dsv4", "glm"):
        relative = f"overlays/{model}/manifest.json"
        if relative not in records:
            continue
        path = review / relative
        overlay = json.loads(path.read_text())
        for row in overlay["files"]:
            row["after_sha256"] = records[f"overlays/{model}/{row['path']}"]["sha256"]
            if model == "glm" and row["path"].startswith("lmcache/"):
                row["before_sha256"] = records.get(row["path"], {}).get("sha256")
        payload = json.dumps(overlay, indent=2).encode()
        path.write_bytes(payload)
        records[relative].update(
            sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)
        )
    return changed


def finalize(repo: Path, review: Path) -> dict:
    """Refresh owned submission tools and verify all other copied source hashes.

    Args:
        repo: Working checkout containing tools/submission.
        review: Previously generated review tree with SOURCE_MANIFEST.json.
    Returns:
        Source hash/syntax validation summary, not GPU execution evidence.
    Raises:
        ValueError: An audited file changed outside the submission-tool refresh.
        SyntaxError: Any copied Python source is invalid.
    """
    manifest_path = review / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    records = {
        row["path"]: row
        for row in manifest["files"]
        if (
            Path(row["path"]).suffix not in {".md", ".rst", ".html", ".pdf", ".docx"}
            or row["path"] == "README.md"
        )
        and row["path"]
        not in {
            "lmcache/v1/metadata_upload.py",
            "tools/submission/metadata_upload.py",
            "scripts/apply_vllm_0260_lmcache.py",
            "scripts/vllm_patches/deepseek_v4_attention.py",
            "scripts/vllm_patches/sparse_attn_indexer.py",
        }
    }
    for source in sorted((repo / "tools/submission").iterdir()):
        if source.suffix != ".py" or not source.is_file():
            continue
        relative = "tools/submission/" + source.name
        shutil.copyfile(source, review / relative)
        payload = source.read_bytes()
        records[relative] = dict(
            path=relative,
            origin="local_worktree",
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
    readme = repo / "README.md"
    shutil.copyfile(readme, review / "README.md")
    records["README.md"] = dict(
        path="README.md",
        origin="submission_readme",
        sha256=hashlib.sha256(readme.read_bytes()).hexdigest(),
        bytes=readme.stat().st_size,
    )
    launcher = repo / "start.sh"
    shutil.copyfile(launcher, review / "start.sh")
    records["start.sh"] = dict(
        path="start.sh",
        origin="submission_launcher",
        sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
        bytes=launcher.stat().st_size,
    )
    ignore = repo / "tools/submission/submission.gitignore"
    shutil.copyfile(ignore, review / ".gitignore")
    records[".gitignore"] = dict(
        path=".gitignore",
        origin="submission_ignore_rules",
        sha256=hashlib.sha256(ignore.read_bytes()).hexdigest(),
        bytes=ignore.stat().st_size,
    )
    # Keep the export tool self-contained in the delivered source tree.
    shutil.copyfile(ignore, review / "tools/submission/submission.gitignore")
    records["tools/submission/submission.gitignore"] = dict(
        records[".gitignore"], path="tools/submission/submission.gitignore"
    )
    attributes = repo / "tools/submission/submission.gitattributes"
    for relative in (".gitattributes", "tools/submission/submission.gitattributes"):
        shutil.copyfile(attributes, review / relative)
        records[relative] = dict(
            path=relative,
            origin="submission_line_endings",
            sha256=hashlib.sha256(attributes.read_bytes()).hexdigest(),
            bytes=attributes.stat().st_size,
        )
    for source_name, relative in (
        ("native_runtime.py", "experiments/legacy/serving_counted_ops_v764.py"),
        ("native_runtime.py", "lmcache/integration/vllm/serving_counted_ops_v764.py"),
        (
            "install_native_runtime.py",
            "experiments/legacy/install_ds_static_runtime_v365.py",
        ),
    ):
        source = repo / "tools/submission" / source_name
        previous = records.get(relative, {})
        payload = source.read_bytes()
        shutil.copyfile(source, review / relative)
        records[relative] = dict(
            path=relative,
            origin="submission_native_startup",
            original_sha256=previous.get("original_sha256", previous.get("sha256")),
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
    syntax_checked = 0
    project = repo / "pyproject.toml"
    project_text = project.read_text(encoding="utf-8")
    if "fallback_version" not in project_text:
        project_text = project_text.replace(
            "[tool.setuptools_scm]\n",
            '[tool.setuptools_scm]\nfallback_version = "0.5.2"\n',
        )
    project_payload = project_text.encode()
    (review / "pyproject.toml").write_bytes(project_payload)
    records["pyproject.toml"] = dict(
        path="pyproject.toml",
        origin="submission_scm_fallback",
        original_sha256=hashlib.sha256(project.read_bytes()).hexdigest(),
        sha256=hashlib.sha256(project_payload).hexdigest(),
        bytes=len(project_payload),
    )
    for relative in CUDA_STREAM_SOURCES:
        original = (repo / relative).read_text(encoding="utf-8")
        payload = normalize_cuda_source(relative, original).encode()
        destination = review / relative
        current = hashlib.sha256(destination.read_bytes()).hexdigest()
        if current not in {
            records[relative]["sha256"],
            hashlib.sha256(payload).hexdigest(),
        }:
            raise ValueError("CUDA audit source changed independently")
        destination.write_bytes(payload)
        records[relative] = dict(
            path=relative,
            origin="submission_cuda_header_fix",
            original_sha256=hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
    for path in sorted((review / "config").glob("*.json")):
        relative = path.relative_to(review).as_posix()
        if relative not in records:
            payload = path.read_bytes()
            records[relative] = dict(
                path=relative,
                origin="submission_configuration",
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            )
    for relative, record in records.items():
        payload = (review / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError(f"Reviewed source changed: {relative}")
        if relative.endswith(".py"):
            ast.parse(payload.decode("utf-8-sig"), filename=relative)
            syntax_checked += 1
    normalized_count = normalize_line_endings(review, records)
    manifest["files"] = list(records.values())
    manifest["status"] = "SOURCE_AUDITED_MODEL_VALIDATION_SEE_README"
    manifest_path.write_bytes(json.dumps(manifest, indent=2).encode())
    (review / "TUTTI_LOCAL.patch").write_bytes(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo / "Tutti"),
                "diff",
                "--no-ext-diff",
                "--",
                "backends",
            ]
        )
    )
    result = dict(
        files=len(records),
        bytes=sum(row["bytes"] for row in records.values()),
        python_syntax_checked=syntax_checked,
        normalized_line_endings=normalized_count,
        all_source_hashes_match=True,
        missing_script_dependencies=manifest["missing_script_dependencies"],
        all_models_gpu_cold_start_validated=False,
        model_validation_scope="See README; this audit command only validates source",
    )
    return result


def main() -> None:
    """Finalize metadata without changing experiment code or publishing anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.repo, args.review)))


if __name__ == "__main__":
    main()
