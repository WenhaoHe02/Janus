# SPDX-License-Identifier: Apache-2.0
"""Prepare strict isolated loader/manager edits for the range-coalescing A/B."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path


HELPER = '''
    def _range_read_v1204(
        self, original_op: Callable[..., None], queue_args: tuple,
        segment_plan: Optional["DeviceSegmentPlan"],
        chunk_ranges: Optional[tuple[tuple[int, int], ...]],
    ) -> None:
        """Route the experimental counted call without changing scatter/events."""
        mode = os.getenv("LMCACHE_READ_RANGE_MODE_V1204", "off")
        if mode == "off":
            original_op(*queue_args)
            return
        if mode not in ("separate", "merge"):
            raise ValueError("Unknown range-read experiment mode")
        if os.getenv("LMCACHE_TUTTI_UNPACKED_SUBMIT_V1139", "0") == "1":
            raise ValueError("Both range arms require identical batched doorbells")
        if os.getenv("LMCACHE_TUTTI_COUNTED_HOLD_GIL", "0") == "1":
            raise ValueError("Range arms preserve baseline GIL-release policy")
        reader = getattr(self, "_range_transport_v1204", None)
        if reader is None:
            import importlib.util
            from read_range_transport_v1204 import ReadRangeTransport
            name = "tutti_read_coalescing_v1203"
            spec = importlib.util.spec_from_file_location(
                name, "/tmp/read_coalescing_v1203/build/tutti_read_coalescing_v1203.so")
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
            info = self._session.info
            reader = ReadRangeTransport(
                native, self._staging, max_data_size=int(info.max_data_size),
                block_size=int(info.block_size), sgl_supported=int(info.sgl_supported),
                merge=mode == "merge", max_members=64,
                verify_readback=os.getenv("LMCACHE_READ_RANGE_VERIFY_V1205", "0") == "1")
            self._range_transport_v1204 = reader
            logger.info("READ_RANGE_V1204_ACTIVE mode=%s max_bytes=%d max_members=%d device=%s",
                        mode, reader.max_bytes, reader.max_members, self._cuda_device)
        if reader.merge != (mode == "merge"):
            raise RuntimeError("Range mode changed after initialization")
        evidence = reader.enqueue(queue_args, chunk_ranges=chunk_ranges, segment_plan=segment_plan)
        if evidence is not None:
            logger.info("READ_RANGE_V1205_READBACK_PASS device=%s evidence=%s",
                        self._cuda_device, evidence)
'''

SNAPSHOT = '''    def read_range_snapshot(self) -> dict:
        """Return cumulative range counters after all counted reads are drained.

        Returns:
            Counter totals, or an empty dict if this loader never used the path.
        Raises:
            RuntimeError: Reads remain in flight; never synchronize them here.
        """
        with self._io_lock:
            if getattr(self, "_device_read_events", []):
                raise RuntimeError("Drain counted reads before taking a snapshot")
            reader = getattr(self, "_range_transport_v1204", None)
            return reader.snapshot_after_drain() if reader is not None else {}

'''


def offsets(source: str, node: ast.AST) -> tuple[int, int]:
    """Return character offsets for a Python AST node's UTF-8 column positions."""
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1]))
    end = sum(map(len, lines[:node.end_lineno - 1]))
    start += len(lines[node.lineno - 1].encode('utf-8')[:node.col_offset].decode('utf-8'))
    end += len(lines[node.end_lineno - 1].encode('utf-8')[:node.end_col_offset].decode('utf-8'))
    return start, end


def transform_loader(source: str) -> str:
    """Add an opt-in counted-read transport, preserving original scatter code.

    Raises ValueError if the reviewed single read-op call or ABI differs.
    """
    if '_range_read_v1204' in source:
        raise ValueError('Already patched')
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TuttiDirectLoader']
    if len(classes) != 1:
        raise ValueError('Loader class changed')
    cls = classes[0]
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name == 'enqueue_device_counted_read']
    if len(methods) != 1:
        raise ValueError('Counted entry changed')
    method = methods[0]
    calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'read_op']
    if len(calls) != 1 or len(calls[0].args) != 20 or calls[0].keywords:
        raise ValueError('Counted native ABI changed')
    call = calls[0]
    arguments = ', '.join(ast.get_source_segment(source, arg) for arg in call.args)
    a, b = offsets(source, call)
    method_start, method_end = offsets(source, method)
    old = source[method_start:method_end]
    relative_start, relative_end = a - method_start, b - method_start
    updated = old[:relative_start] + (
        f'self._range_read_v1204(read_op, ({arguments},), segment_plan, coalescing_chunk_ranges)'
    ) + old[relative_end:]
    anchor = '        packed_entries_per_group: int = 0,\n'
    if updated.count(anchor) != 1:
        raise ValueError('Counted signature changed')
    updated = updated.replace(anchor, anchor +
        '        coalescing_chunk_ranges: Optional[tuple[tuple[int, int], ...]] = None,\n')
    doc_anchor = '            packed_entries_per_group: Positive for a 3-D entry-slot cache;'
    if doc_anchor not in updated:
        raise ValueError('Counted API docstring changed')
    updated = updated.replace(doc_anchor,
        '            coalescing_chunk_ranges: Immutable source-chunk boundaries for\n'
        '                the isolated range experiment; None preserves normal reads.\n' + doc_anchor)
    _, class_end = offsets(source, cls)
    # Reverse-order replacements keep AST offsets valid. Public getter goes
    # before the enqueue method; new private implementation goes at class end.
    result = source[:class_end] + '\n' + HELPER + source[class_end:]
    result = result[:method_start] + SNAPSHOT.lstrip() + '    ' + updated + result[method_end:]
    # Both arms collect identical cumulative counters at an existing drain.
    # This is lightweight instrumentation, not a claim of uninstrumented TTFT.
    finished = [n for n in ast.walk(ast.parse(result)) if isinstance(n, ast.FunctionDef)
                and n.name == 'finish_device_counted_reads']
    if len(finished) != 1:
        raise ValueError('Counted drain method changed')
    clears = [n for n in ast.walk(finished[0]) if isinstance(n, ast.Expr)
              and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
              and isinstance(n.value.func.value, ast.Subscript)
              and isinstance(n.value.func.value.value, ast.Name)
              and n.value.func.value.value.id == 'events' and n.value.func.attr == 'synchronize']
    if len(clears) != 1:
        raise ValueError('Counted drain anchor changed')
    _, end = offsets(result, clears[0])
    result = result[:end] + '''
            with self._io_lock:
                reader = getattr(self, "_range_transport_v1204", None)
                if reader is not None and not getattr(self, "_device_read_events", []):
                    logger.info("READ_RANGE_V1205_COUNTS device=%s totals=%s",
                                self._cuda_device, reader.snapshot_after_drain())
''' + result[end:]
    compile(result, 'range_loader_v1204', 'exec')
    return result


def transform_manager(source: str) -> str:
    """Pass immutable source-chunk boundaries to all three counted call sites.

    Grouped HCA/native-indexer reads use their actual group count, not CSA block
    indices. Normal runtime mode leaves the argument None.
    Raises ValueError when a caller or its required local geometry has changed.
    """
    if 'coalescing_chunk_ranges=' in source:
        raise ValueError('Already patched')
    edits = []
    names = set()
    for method in (node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(method):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'enqueue_device_counted_read'
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'loader'):
                continue
            names.add(method.name)
            a, b = offsets(source, node)
            old = source[a:b]
            prefix, tail = old.rsplit('\n', 1)
            if tail.strip() != ')' or not prefix.rstrip().endswith(','):
                raise ValueError('Unexpected call formatting')
            if method.name == '_enqueue_aligned_group_read':
                expression = '((0, (entries + group_entries - 1) // group_entries),)'
            elif method.name == '_enqueue_device_grouped_layer_read':
                expression = '((0, entries // group_entries),)'
            else:
                expression = ('tuple((chunk.first_compressed_block, chunk.end_compressed_block) '
                              'for chunk in state.chunks)')
            value = (expression + ' if os.getenv("LMCACHE_READ_RANGE_MODE_V1204", "off") '
                     'in ("separate", "merge") else None')
            indent = tail[:len(tail) - len(tail.lstrip())]
            updated = prefix + '\n' + indent + '    coalescing_chunk_ranges=' + value + ',\n' + tail
            edits.append((a, b, updated))
    expected_names = {'try_device_miss_correction', '_enqueue_device_owner_reads',
                      '_enqueue_device_grouped_layer_read'}
    if len(edits) == 5:
        expected_names |= {'_enqueue_device_owner_ids', '_enqueue_aligned_group_read'}
    if len(edits) not in (3, 5) or names != expected_names:
        raise ValueError(f'Expected correction/owner/grouped callers, got {names}')
    result = source
    for a, b, updated in sorted(edits, reverse=True):
        result = result[:a] + updated + result[b:]
    compile(result, 'range_manager_v1204', 'exec')
    return result


def main() -> None:
    """Generate reviewable candidate files, never patch an active installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    proof = {}
    for relative, transform in (
        ('v1/gpu_connector/tutti_direct_loader.py', transform_loader),
        ('v1/csa_attention_kv_prefetch_manager.py', transform_manager),
    ):
        source = (args.lmcache_root / relative).read_text()
        candidate = transform(source)
        target = args.output / Path(relative).name
        target.write_text(candidate)
        proof[relative] = dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                               candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest())
    (args.output / 'SOURCES.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(dict(prepared=True, installed=False, sources=proof)))


if __name__ == '__main__':
    main()
