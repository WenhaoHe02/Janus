# SPDX-License-Identifier: Apache-2.0
"""Assemble measured runtime source and remove request-bypassing experiments."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

# Local
from install_native_runtime import BINDING


def isolate_handoff_commands(source: str) -> str:
    """Keep profiler preload threads out of nsenter's single-threaded setns call."""
    tree = ast.parse(source)
    for function in tree.body:
        if (
            not isinstance(function, ast.FunctionDef)
            or function.name != "_maybe_unmount_for_tutti"
        ):
            continue
        original = ast.get_source_segment(source, function)
        if "handoff_env" in original:
            return source
        replacement = original.replace(
            "            host_umount = subprocess.run(",
            """            # Profiler injection can start helper threads in nsenter;
            # setns(CLONE_NEWNS) rejects a shared filesystem context with EINVAL.
            handoff_env = {
                key: value for key, value in os.environ.items()
                if key not in {
                    "LD_PRELOAD", "LD_AUDIT", "NVTX_INJECTION64_PATH",
                    "CUDA_INJECTION64_PATH",
                }
            }
            host_umount = subprocess.run(""",
        )
        replacement, count = re.subn(
            r"(host_(?:umount|still_mounted) = subprocess.run\([\s\S]*?)(check=False,)",
            r"\1env=handoff_env,\n                \2",
            replacement,
        )
        if count != 2:
            raise ValueError("Unknown host mount handoff calls")
        return source.replace(original, replacement, 1)
    raise ValueError("Missing host mount handoff function")


def fail_closed_loader_startup(source: str) -> str:
    """Do not initialize or hand off SSDs after a model-health timeout."""
    tree = ast.parse(source)
    insertions = []
    for function in ast.walk(tree):
        if (
            not isinstance(function, ast.FunctionDef)
            or function.name != "_startup_warmup"
        ):
            continue
        for node in ast.walk(function):
            if (
                not isinstance(node, ast.If)
                or not isinstance(node.test, ast.Name)
                or node.test.id != "ready"
            ):
                continue
            if not node.orelse or isinstance(node.orelse[-1], ast.Return):
                continue
            last = node.orelse[-1]
            if "health poll timed out" in ast.get_source_segment(source, last):
                insertions.append((last.end_lineno, last.col_offset))
    lines = source.splitlines(keepends=True)
    for line, indent in sorted(insertions, reverse=True):
        lines.insert(line, " " * indent + "return\n")
    return "".join(lines).replace(
        "proceeding anyway (worker=%d)", "not initializing loader (worker=%d)"
    )


def normalize_runtime(relative: str, source: str, native_helper: str) -> str | None:
    """Remove only known decode bypasses/backups and replace native startup binding.

    Returns a syntax-checked source or None for excluded implementation backups.
    Raises ValueError if any unhandled bypass or obsolete binary path remains.
    """
    source = source.replace("\r\n", "\n")
    if ".before_" in relative or relative.endswith("serving_decode_policy_v763.py"):
        return None
    if relative.endswith("serving_counted_ops_v764.py"):
        return native_helper
    if relative == "lmcache/v1/cache_engine.py":
        source = fail_closed_loader_startup(source)
        source = isolate_handoff_commands(source)
    if relative == "lmcache/__init__.py":
        anchor = '    sys.modules["lmcache.c_ops"] = _ops\n'
        if anchor in source and "    c_ops = _ops\n" not in source:
            source = source.replace(anchor, anchor + "    c_ops = _ops\n")
    import_line = (
        "from lmcache.integration.vllm.serving_decode_policy_v763 "
        "import skip_decode_reads\n"
    )
    ds_guard = (
        "    # Serving v889: explicit idealized decode, never infer from Q length.\n"
        "    "
        + import_line
        + '    if skip_decode_reads(phases[request] == "decode"):\n'
        '        manager.set_kv_forward_phase("inactive")\n'
        "        return False\n"
    )
    glm_guard = (
        "    if skip_decode_reads(glm_forward_phase.is_decode()):\n"
        "        return False\n"
    )
    source = (
        source.replace(ds_guard, "").replace(glm_guard, "").replace(import_line, "")
    )
    old_marker = "# DS_STATIC_COUNTED_RUNTIME_V365:"
    if old_marker in source:
        start = source.index(old_marker)
        end = source.index("class TuttiDirectLoader:", start)
        source = source[:start] + BINDING + source[end:]
    if "skip_decode_reads" in source or "ASSUMED_HBM" in source:
        raise ValueError(f"Unhandled decode bypass in {relative}")
    if "/tmp/device_count_v137/build/tutti_device_count_v137.so" in source:
        raise ValueError(f"Obsolete native loader in {relative}")
    ast.parse(source, filename=relative)
    return source


def assemble(archive_path: Path, receipt_path: Path, review: Path, tools: Path) -> dict:
    """Build DS default sources and GLM/vLLM overlays from verified container bytes.

    Args:
        archive_path: Source-only collector zip.
        receipt_path: Collector receipt identifying the captured deployments.
        review: Existing, private submission review tree.
        tools: Directory containing native_runtime.py.
    Returns:
        Model source provenance and overlay file manifests.
    Raises:
        ValueError: Archive digest, Flash/Pro equality or source safety differs.
    The experimental containers and original local checkout are never modified.
    """
    provenance = json.loads(receipt_path.read_text())
    if (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != provenance["archive_sha256"]
    ):
        raise ValueError("Source archive checksum mismatch")
    if provenance["source_hashes"]["flash"] != provenance["source_hashes"]["pro"]:
        raise ValueError("Flash and Pro require separate source selections")
    manifest_path = review / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    records = {row["path"]: row for row in manifest["files"]}
    native = (tools / "native_runtime.py").read_text()
    desired = {}
    overlays = {"dsv4": [], "glm": []}
    with zipfile.ZipFile(archive_path) as archive:
        base = {
            name[5:]: archive.read(name)
            for name in archive.namelist()
            if name.startswith("base/")
        }
        sources = {}
        for model in ("flash", "glm"):
            current = dict(base)
            for relative in provenance["deltas"][model]["changed"]:
                payload = archive.read(model + "/" + relative)
                if (
                    hashlib.sha256(payload).hexdigest()
                    != provenance["source_hashes"][model][relative]
                ):
                    raise ValueError("Container source checksum mismatch")
                current[relative] = payload
            normalized = {}
            for relative, payload in current.items():
                if (
                    not relative.startswith("lmcache/")
                    and relative not in provenance["deltas"][model]["changed"]
                ):
                    continue
                source = normalize_runtime(
                    relative, payload.decode("utf-8-sig"), native
                )
                if source is not None:
                    normalized[relative] = source.encode()
            normalized["lmcache/integration/vllm/serving_counted_ops_v764.py"] = (
                native.encode()
            )
            sources[model] = normalized
        for relative, payload in sources["flash"].items():
            if relative.startswith("lmcache/"):
                desired[relative] = (payload, "flash")
            else:
                destination = "overlays/dsv4/" + relative
                desired[destination] = (payload, "flash")
                overlays["dsv4"].append(
                    dict(
                        path=relative,
                        before_sha256=hashlib.sha256(base[relative]).hexdigest(),
                        after_sha256=hashlib.sha256(payload).hexdigest(),
                    )
                )
        for relative, payload in sources["glm"].items():
            if relative.startswith("vllm/") or payload != sources["flash"].get(
                relative
            ):
                destination = "overlays/glm/" + relative
                desired[destination] = (payload, "glm")
                prior = (
                    sources["flash"].get(relative)
                    if relative.startswith("lmcache/")
                    else base.get(relative)
                )
                overlays["glm"].append(
                    dict(
                        path=relative,
                        before_sha256=hashlib.sha256(prior).hexdigest()
                        if prior is not None
                        else None,
                        after_sha256=hashlib.sha256(payload).hexdigest(),
                    )
                )
    # Validate all destinations before making the first write.
    root = review.resolve()
    for relative in desired:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError("Unsafe output path")
        if path.exists() and relative in records:
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != records[relative]["sha256"]
            ):
                raise ValueError(f"Audit copy was edited independently: {relative}")
    for relative, (payload, model) in desired.items():
        path = review / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records[relative] = dict(
            path=relative,
            origin="captured_runtime_normalized",
            container_id=provenance["models"][model]["container_id"],
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
    for model, rows in overlays.items():
        path = review / "overlays" / model / "manifest.json"
        path.write_text(json.dumps(dict(model=model, files=rows), indent=2))
        relative = path.relative_to(review).as_posix()
        payload = path.read_bytes()
        records[relative] = dict(
            path=relative,
            origin="overlay_configuration",
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )
    manifest["files"] = list(records.values())
    manifest["runtime_models"] = provenance["models"]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return dict(
        written=len(desired),
        dsv4_overlay=len(overlays["dsv4"]),
        glm_overlay=len(overlays["glm"]),
        flash_pro_sources_identical=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.archive, args.receipt, args.review, args.tools)))
