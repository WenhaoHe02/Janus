# SPDX-License-Identifier: Apache-2.0
"""Create a source-only review tree without changing the working checkout."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

# Local
from release_decode_cleanup import submission_source

SOURCE_SUFFIXES = {".py", ".sh", ".cu", ".cuh", ".cpp", ".c", ".h", ".hpp"}
ROOT_FILES = (
    "start.sh",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "setup.py",
    "MANIFEST.in",
    "CMakeLists.txt",
    ".clang-format",
    ".isort.cfg",
)
EXPERIMENT_ROOTS = (
    "run_original_serving_v957.py",
    "serving_ds_configuration_v890.py",
    "serving_glm_configuration_v766.py",
    "startup_ds_serving_v916.sh",
    "startup_glm_serving_v766.sh",
    "warmup_actual_requests_v1073.py",
    "prewarm_glm_proxy_buckets_v438.py",
    "np_write_modes_v1220.py",
    "install_np_write_modes_v1220.py",
    "np_write_modes_configuration_v1221.py",
    "prepare_flash_ttft_ablation_v1185.py",
    "read_range_coalescing_v1203.cu",
    "install_read_range_v1204.py",
    "read_range_plan_v1202.py",
    "run_np_concurrent_writes_v1223.py",
    "run_glm_layerwise_periodic_v1199.py",
    "test_read_range_plan_v1202.py",
    "test_actual_warmup_v1088.py",
    "test_serving_glm_configuration_v766.py",
    "build_read_coalescing_v1203.py",
    "build_unpacked_submit_v1139.py",
    "test_build_read_coalescing_v1203.py",
    "build_device_count_v137.py",
)


def git(repo: Path, *args: str) -> str:
    """Read Git metadata from repo, raising on any Git failure."""
    return subprocess.check_output(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
    ).strip()


def source_dependencies(source: str, suffix: str) -> set[str]:
    """Find local-looking imports and literal /tmp script references.

    Returns candidate basenames, not a proof of dynamic dependency closure.
    Syntax errors in Python raise SyntaxError; caller must not hide them.
    """
    result = set(re.findall(r"/tmp/([\w.-]+\.(?:py|sh|cu|cuh))\b", source))
    result.update(re.findall(r"['\"]([\w]+_v\d+\.py)['\"]", source))
    if suffix == ".py":
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                first = name.split(".")[0]
                if re.search(r"_v\d+$", first) or first.startswith("run_"):
                    result.add(first + ".py")
    return result


def build(repo: Path, output: Path, extra: Path | None = None) -> dict:
    """Copy allowlisted source into a new directory, with provenance and risks.

    Args:
        repo: Existing checkout, including its uncommitted source changes.
        output: New, absent audit directory; never overwritten.
        extra: Optional directory of recovered remote source files.
    Returns:
        Manifest with hashes, missing dependencies, and manual-review risks.
    Raises:
        FileExistsError: Output already exists.
        ValueError: A source is a symlink or contains a likely secret.
    """
    repo, output = repo.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, risks, missing = [], [], set()

    def copy(source: Path, relative: str, origin: str) -> None:
        """Copy one regular text file and record its source digest."""
        if source.is_symlink():
            raise ValueError(f"Symlink requires explicit audit: {relative}")
        payload = source.read_bytes()
        text = payload.decode("utf-8-sig")
        original_digest = hashlib.sha256(payload).hexdigest()
        cleaned = submission_source(relative, text)
        if relative == "pyproject.toml" and "fallback_version" not in text:
            cleaned = text.replace(
                "[tool.setuptools_scm]\n",
                '[tool.setuptools_scm]\nfallback_version = "0.5.2"\n',
            )
        native_templates = {
            "experiments/legacy/serving_counted_ops_v764.py": "native_runtime.py",
            "experiments/legacy/install_ds_static_runtime_v365.py": (
                "install_native_runtime.py"
            ),
        }
        if relative in native_templates:
            cleaned = (
                repo / "tools/submission" / native_templates[relative]
            ).read_text()
        if cleaned is None:
            return
        changed = cleaned != text
        if changed:
            text = cleaned
            payload = cleaned.encode("utf-8")
        if re.search(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----", text):
            raise ValueError(f"Private key in {relative}")
        # Report assignments, never print their values into audit output.
        credentials = re.findall(
            r"(?im)^\s*(?:export\s+)?(?:password|passwd|api_key)\s*=\s*['\"]([^'\"]+)['\"]",
            text,
        )
        if any(value not in {"sk-dummy", "EMPTY", "dummy"} for value in credentials):
            raise ValueError(
                f"Possible literal credential requires removal: {relative}"
            )
        if re.search(r"\b(?:sudo|mkfs|umount|rmmod)\b|rm\s+-rf", text):
            risks.append(
                dict(file=relative, kind="host_or_destructive_operation_review")
            )
        if "/tmp/" in text or "/pro_model" in text or "172.16." in text:
            risks.append(dict(file=relative, kind="machine_specific_path_or_address"))
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if changed:
            target.write_bytes(payload)
        else:
            shutil.copyfile(source, target)
        records.append(
            dict(
                path=relative,
                origin=origin,
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                original_sha256=original_digest,
                submission_transform="release_normalization" if changed else None,
            )
        )

    files = git(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "lmcache",
        "csrc",
        "rust",
        "requirements",
        "tools/submission",
    ).splitlines()
    for relative in sorted(set(files) | set(ROOT_FILES)):
        source = repo / relative
        if source.suffix in {".md", ".rst", ".html", ".pdf", ".docx"}:
            if relative != "README.md":
                continue
            source = repo / "tools/submission/README.md"
        if not source.is_file():
            continue
        if (
            source.suffix
            not in SOURCE_SUFFIXES | {".rs", ".toml", ".txt", ".lock", ".md"}
            and relative not in ROOT_FILES
        ):
            continue
        if any(part in {"__pycache__", "target", "build"} for part in source.parts):
            continue
        copy(source, relative, "local_worktree")
    pending, seen = list(EXPERIMENT_ROOTS), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        candidates = [repo / "experiments" / "legacy" / name]
        if extra is not None:
            candidates.insert(0, extra / name)
        source = next((p for p in candidates if p.is_file()), None)
        if source is None:
            missing.add(name)
            continue
        text = source.read_text(encoding="utf-8-sig")
        pending.extend(source_dependencies(text, source.suffix) - seen)
        copy(
            source,
            "experiments/legacy/" + name,
            "remote_recovered"
            if extra and source.parent == extra
            else "local_experiment_source",
        )

    native_root = extra / "device_count_v137" if extra else None
    for name in (
        "csrc/tutti_kv_ops.cu",
        "csrc/tutti_kv_ops.cuh",
        "device_count_bindings_v137.cpp",
    ):
        if native_root is None or not (native_root / name).is_file():
            missing.add("device_count_v137/" + name)
        else:
            copy(
                native_root / name,
                "experiments/native/device_count_v137/" + name,
                "remote_native_source",
            )

    manifest = dict(
        schema=1,
        status="AUDIT_CANDIDATE_NOT_GPU_VALIDATED",
        head=git(repo, "rev-parse", "HEAD"),
        dev_base=git(repo, "rev-parse", "dev"),
        includes_uncommitted_source=True,
        files=records,
        missing_script_dependencies=sorted(missing),
        manual_review=risks,
        excluded=[
            "results",
            "private traces",
            "tokens",
            "weights",
            "logs",
            "compiled binaries",
            "SSH configuration",
            "Git history",
            "unselected temporary scripts",
            "Tutti upstream source",
        ],
        limitations=[
            "Static closure does not resolve generated/dynamic imports",
            "Local source is not asserted identical to deployed overlays",
            "Legacy launchers are retained for audit, not portable deployment",
        ],
    )
    (output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Build an audit candidate; never commit, push, or launch experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=Path)
    args = parser.parse_args()
    manifest = build(args.repo, args.output, args.extra)
    print(
        json.dumps(
            dict(
                files=len(manifest["files"]),
                missing=manifest["missing_script_dependencies"],
                manual_review=len(manifest["manual_review"]),
            )
        )
    )


if __name__ == "__main__":
    main()
