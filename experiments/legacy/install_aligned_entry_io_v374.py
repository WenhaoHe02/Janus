# SPDX-License-Identifier: Apache-2.0
"""Install opt-in unaligned grouped entry transport in a new container only."""

# Standard
import argparse
import ast
from pathlib import Path


def method_span(source: str, class_name: str, method: str) -> tuple[int, int]:
    """Return source offsets for the uniquely named method definition."""
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )
    lines = source.splitlines(keepends=True)
    return sum(map(len, lines[: node.lineno - 1])), sum(
        map(len, lines[: node.end_lineno])
    )


def replace_once(source: str, old: str, new: str) -> str:
    """Apply one checked anchor replacement, rejecting a different code layout."""
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one source anchor: {old!r}")
    return source.replace(old, new, 1)


def install(root: Path) -> None:
    """Extend only grouped HCA/native-entry transport, retaining the old default.

    Args:
        root: LMCache package in the newly created experiment container.
    Raises:
        RuntimeError: Frozen method anchors do not match the supported version.
    """
    loader_path = root / "v1/gpu_connector/tutti_direct_loader.py"
    source = loader_path.read_text()
    if "ALIGNED_ENTRY_IO_V374" not in source:
        start, end = method_span(
            source, "TuttiDirectLoader", "enqueue_device_counted_read"
        )
        body = source[start:end]
        body = replace_once(
            body,
            "        packed_entries_per_group: int = 0,\n",
            "        packed_entries_per_group: int = 0,\n"
            "        entry_copy_lengths: Optional[torch.Tensor] = None,\n"
            "        entry_source_offsets: Optional[torch.Tensor] = None,\n",
        )
        body = replace_once(
            body,
            "        queued = False\n",
            """        # ALIGNED_ENTRY_IO_V374: validate before any NVMe submission.
        if entry_copy_lengths is not None or entry_source_offsets is not None:
            if (not packed_entries or segment_plan is None
                    or entry_copy_lengths is None or entry_source_offsets is None):
                raise ValueError("aligned entry copies require grouped segment tables")
            extra_tables = (entry_copy_lengths, entry_source_offsets)
            if any(not t.is_cuda or t.device != destination.device
                   or t.dtype != torch.int32 or t.ndim != 1
                   or t.numel() != segment_plan.logical_ids.numel()
                   or not t.is_contiguous() for t in extra_tables):
                raise ValueError("invalid aligned entry copy tables")
            tensors += extra_tables
        queued = False
""",
        )
        # Move validation before reader reservation so malformed tables cannot
        # leak the reader count before entering the existing try/finally.
        begin = body.index("        # ALIGNED_ENTRY_IO_V374")
        finish = body.index("        queued = False\n", begin)
        validation = body[begin:finish]
        body = body[:begin] + body[finish:]
        body = replace_once(
            body,
            "\n        with self._reader_gate:\n",
            "\n" + validation + "        with self._reader_gate:\n",
        )
        body = replace_once(
            body,
            '                    scatter_kwargs = {"segment_plan": segment_plan}\n',
            """                    scatter_kwargs = {"segment_plan": segment_plan}
                    if entry_copy_lengths is not None:
                        from lmcache.v1.aligned_entry_scatter_v374 import (
                            scatter_aligned_device_entries,
                        )
                        scatter = scatter_aligned_device_entries
                        scatter_kwargs["entry_copy_lengths"] = entry_copy_lengths
                        scatter_kwargs["entry_source_offsets"] = entry_source_offsets
""",
        )
        source = source[:start] + body + source[end:]
        compile(source, str(loader_path), "exec")
        loader_path.write_text(source)
    manager_path = root / "v1/csa_attention_kv_prefetch_manager.py"
    source = manager_path.read_text()
    if "def _enqueue_aligned_group_read(" not in source:
        start, end = method_span(
            source,
            "CSAAttentionKVPrefetchManager",
            "_enqueue_device_grouped_layer_read",
        )
        body = source[start:end]
        body = replace_once(
            body,
            "        chunk = state.chunks[0]\n",
            """        if (os.getenv("LMCACHE_DSV4_ALIGNED_ENTRY_IO", "0") == "1"
                and (state.chunks[0].n_compressed_blocks % group_entries
                     or state.chunks[0].layer_byte_offset % 512)):
            return self._enqueue_aligned_group_read(state, group_entries, destination)
        chunk = state.chunks[0]
""",
        )
        source = source[:start] + body + source[end:]
        start, end = method_span(
            source,
            "CSAAttentionKVPrefetchManager",
            "_enqueue_device_native_indexer_read",
        )
        body = source[start:end]
        body = replace_once(
            body,
            "        if not group_rows:\n",
            """        if (not group_rows
                and os.getenv("LMCACHE_DSV4_ALIGNED_ENTRY_IO", "0") == "1"):
            group_rows = 2
        if not group_rows:
""",
        )
        source = source[:start] + body + source[end:]
        # Insert inside this class, even if the module later gains other classes.
        cls = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "CSAAttentionKVPrefetchManager"
        )
        lines = source.splitlines(keepends=True)
        position = sum(map(len, lines[: cls.end_lineno]))
        source = (
            source[:position]
            + Path("/tmp/aligned_group_method_v374.txt").read_text()
            + source[position:]
        )
        compile(source, str(manager_path), "exec")
        manager_path.write_text(source)
    for filename in ("aligned_entry_io_v374.py", "aligned_entry_scatter_v374.py"):
        (root / "v1" / filename).write_bytes((Path("/tmp") / filename).read_bytes())
    print("ALIGNED_ENTRY_IO_V374_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)
