# SPDX-License-Identifier: Apache-2.0
"""Prepare guarded per-block executor hooks in the actual NP raw writer."""

# Standard
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def replace_once(source: str, before: str, after: str) -> str:
    """Replace a unique reviewed anchor or reject the unrecognized source."""
    if source.count(before) != 1:
        raise ValueError(f'Writer source changed: {before[:100]}')
    return source.replace(before, after)


def transform_writer(source: str) -> str:
    """Add an opt-in executor without changing default CPU copy/write semantics.

    Arguments and return type are source strings. Raises ValueError on a
    changed source or syntax error. This function does not install anything.
    """
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CPURawBlockWriter')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    names = [arg.arg for arg in init.args.kwonlyargs]
    if names != ['target_mib_s', 'block_bytes', 'wait_for_admission', 'write_io_scope']:
        raise ValueError('Unexpected raw writer constructor contract')
    # Earlier installers ast.unparse this signature; support both source layouts.
    init.args.kwonlyargs.extend([
        ast.arg(arg='block_executor', annotation=ast.parse('Callable | None', mode='eval').body),
        ast.arg(arg='block_executor_close', annotation=ast.parse('Callable[[], None] | None', mode='eval').body),
    ])
    init.args.kw_defaults.extend([ast.Constant(None), ast.Constant(None)])
    lines = source.splitlines(keepends=True)
    lines[init.lineno - 1:init.body[0].lineno - 1] = [
        '    def __init__(' + ast.unparse(init.args) + ') -> None:\n']
    source = ''.join(lines)
    source = replace_once(source, '        self._write_io_scope = write_io_scope\n',
                          '        self._write_io_scope = write_io_scope\n'
                          '        self._block_executor = block_executor\n'
                          '        self._block_executor_close = block_executor_close\n')
    source = replace_once(source, '            self._buffer.close()\n',
                          '            if self._block_executor_close is not None:\n'
                          '                self._block_executor_close()\n'
                          '            self._buffer.close()\n')
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CPURawBlockWriter')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'write')
    scopes = [n for n in ast.walk(method) if isinstance(n, ast.With)
              and len(n.items) == 1 and isinstance(n.items[0].context_expr, ast.Name)
              and n.items[0].context_expr.id == 'scope']
    if len(scopes) != 1:
        raise ValueError('Original admission scope not unique')
    scope = scopes[0]
    lines = source.splitlines(keepends=True)
    begin, end = scope.body[0].lineno - 1, scope.end_lineno
    original = lines[begin:end]
    indent = ' ' * scope.body[0].col_offset
    replacement = [
        indent + 'if self._block_executor is not None:\n',
        indent + '    block_started_s = time.perf_counter()\n',
        indent + '    written = self._block_executor(\n',
        indent + '        self._fd, self._buffer, payload_view, source_offset,\n',
        indent + '        source_nbytes, io_nbytes, slba * _LBA_BYTES + extent_cursor)\n',
        indent + 'else:\n',
    ] + ['    ' + line for line in original]
    lines[begin:end] = replacement
    result = ''.join(lines)
    compile(result, 'np_raw_writer_with_executor', 'exec')
    return result


FACTORY = '''    def np_write_block_executor_v1220(self):
        """Return a real, opt-in NP write executor; CPU default is unchanged."""
        mode = os.environ.get("LMCACHE_NP_WRITE_MODE_V1220", "cpu")
        if mode == "cpu":
            return None
        if mode not in {"gpu_pack_cpu_write", "gpu_nvme_write"}:
            raise ValueError("Invalid NP write mode")
        if getattr(self, "_np_write_helper_v1220", None) is None:
            from lmcache.v1.storage_backend.np_write_modes_v1220 import (
                MappedGpuBlockCopy, GpuNvmeBlockWrite,
            )
            if mode == "gpu_pack_cpu_write":
                helper = MappedGpuBlockCopy(self._cpu_raw_writer_block_bytes, self._cuda_device)
            else:
                q = self._session.queue
                helper = GpuNvmeBlockWrite(
                    cuda_device=self._cuda_device, capacity=self._cpu_raw_writer_block_bytes,
                    staging=self._staging, stream=self._io_stream, io_lock=self._io_lock,
                    page_iovas=self._session.staging_iovas,
                    queue=dict(sq=q.sq_tensor.data_ptr(), cq=q.cq_tensor.data_ptr(),
                        sq_db=self._session.db_gpu_ptr(q.sq_db_offset),
                        cq_db=self._session.db_gpu_ptr(q.cq_db_offset),
                        tail=self._sq_tail_ptr, head=self._cq_head_ptr,
                        phase=self._cq_phase_ptr, timeout=self._timed_out_ptr,
                        depth=self._q_depth(), qid=q.qid, nsid=self._session.nsid))
            self._np_write_helper_v1220 = helper
        return self._np_write_helper_v1220.execute

    def close_np_write_executor_v1220(self) -> None:
        """Close experiment-owned buffers without unmapping loader-owned resources."""
        helper = getattr(self, "_np_write_helper_v1220", None)
        if helper is not None:
            helper.close()
            self._np_write_helper_v1220 = None

'''


def transform_loader(source: str) -> str:
    """Wire the public writer executor at its unique actual NP construction site."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TuttiDirectLoader')
    calls = [n for n in ast.walk(cls) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'CPURawBlockWriter']
    if len(calls) != 1 or 'np_write_block_executor_v1220' in source:
        raise ValueError('Unrecognized or already patched loader')
    call = calls[0]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:call.lineno - 1])) + call.col_offset
    end = sum(map(len, lines[:call.end_lineno - 1])) + call.end_col_offset
    original = source[start:end]
    replacement = original[:-1].rstrip().rstrip(',') + (
        ', block_executor=self.np_write_block_executor_v1220(), '
        'block_executor_close=self.close_np_write_executor_v1220)')
    result = source[:start] + replacement + source[end:]
    result = replace_once(result, '    def _ensure_cpu_raw_writer(', FACTORY + '    def _ensure_cpu_raw_writer(')
    # Some deployments filter backend profile warnings. Preserve first failure
    # detail on stderr so failed waves cannot hide behind successful publication.
    result = replace_once(result,
        '        except Exception:\n            self.finish_background_write(\n',
        '        except Exception as write_error:\n'
        '            if not getattr(self, "_np_write_error_logged_v1220", False):\n'
        '                self._np_write_error_logged_v1220 = True\n'
        '                import traceback\n'
        '                os.write(2, ("NP_WRITE_V1220_FIRST_FAILURE " + repr(write_error) + "\\n" + traceback.format_exc()).encode())\n'
        '            self.finish_background_write(\n')
    compile(result, 'np_write_loader', 'exec')
    return result


def main() -> None:
    """Validate both files first, then install opt-in sources before worker imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-root', type=Path, required=True)
    args = parser.parse_args()
    pending = []
    proof = {}
    for relative, transform in (
        ('v1/storage_backend/cpu_raw_writer.py', transform_writer),
        ('v1/gpu_connector/tutti_direct_loader.py', transform_loader),
    ):
        path = args.lmcache_root / relative
        source = path.read_text()
        candidate = transform(source)
        backup = path.with_suffix('.py.before_np_write_v1220')
        if backup.exists():
            raise FileExistsError(backup)
        pending.append((path, backup, source, candidate))
        proof[relative] = dict(before_sha256=hashlib.sha256(source.encode()).hexdigest(),
                               after_sha256=hashlib.sha256(candidate.encode()).hexdigest())
    helper = Path('/tmp/np_write_modes_v1220.py')
    compile(helper.read_text(), str(helper), 'exec')
    for path, backup, source, candidate in pending:
        backup.write_text(source)
        path.write_text(candidate)
    shutil.copy2(helper, args.lmcache_root / 'v1/storage_backend/np_write_modes_v1220.py')
    proof['helper_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
    (args.lmcache_root / 'np_write_v1220_provenance.json').write_text(json.dumps(proof, indent=2))
    print('NP_WRITE_V1220_INSTALLED', json.dumps(proof), flush=True)


if __name__ == '__main__':
    main()
