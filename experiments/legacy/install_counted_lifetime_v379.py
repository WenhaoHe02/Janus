# SPDX-License-Identifier: Apache-2.0
"""Honor counted-input keepalive with leases retired only after GPU completion."""

# Standard
import argparse
from pathlib import Path

# Local
from install_aligned_entry_io_v374 import method_span, replace_once


def patch_source(source: str) -> str:
    """Replace per-input allocator tracking with optional completion-held references.

    Args:
        source: Current installed TuttiDirectLoader source, not a frozen archive.
    Returns:
        Patched source; repeated application returns identical text.
    Raises:
        RuntimeError: Expected constructor/enqueue/completion anchors differ.
    Notes:
        The default retains record_stream. With the existing keepalive flag on,
        all input views are retained until successful stream completion. A failed
        completion keeps references attached to the loader for failure teardown.
        No locks, read reservations, CUDA kernels or stream ordering are removed.
    """
    if "COUNTED_LIFETIME_V379" in source:
        return source
    source = replace_once(
        source,
        "        self._device_read_events: list[torch.cuda.Event] = []\n",
        """        self._device_read_events: list[torch.cuda.Event] = []
        # COUNTED_LIFETIME_V379: lifetime spans the asynchronous consumer stream.
        self._counted_input_keepalive = (
            os.getenv("LMCACHE_TUTTI_COUNTED_INPUT_KEEPALIVE", "0") == "1"
        )
        self._device_read_leases: dict[
            torch.cuda.Event, tuple[torch.Tensor, ...]
        ] = {}
""",
    )
    start, end = method_span(source, "TuttiDirectLoader", "enqueue_device_counted_read")
    body = source[start:end]
    body = replace_once(
        body,
        "                    for tensor in tensors:\n"
        "                        tensor.record_stream(stream)\n",
        """                    if self._counted_input_keepalive:
                        self._device_read_leases[done] = tensors
                    else:
                        for tensor in tensors:
                            tensor.record_stream(stream)
""",
    )
    source = source[:start] + body + source[end:]
    start, end = method_span(source, "TuttiDirectLoader", "finish_device_counted_reads")
    body = source[start:end]
    body = replace_once(
        body,
        "            events[-1].synchronize()\n",
        """            events[-1].synchronize()
            # Do not release on synchronization failure, nor retire a new batch
            # enqueued concurrently while the previous FIFO prefix was drained.
            with self._io_lock:
                for event in events:
                    self._device_read_leases.pop(event, None)
""",
    )
    source = source[:start] + body + source[end:]
    start, end = method_span(source, "TuttiDirectLoader", "close")
    body = source[start:end]
    body = replace_once(
        body,
        "            for stream in (self._io_stream, self._host_pack_stream):\n",
        "            streams_quiesced = True\n"
        "            for stream in (self._io_stream, self._host_pack_stream):\n",
    )
    body = replace_once(
        body,
        '                    logger.warning("Tutti CUDA stream '
        'synchronization failed: %s", exc)\n',
        '                    logger.warning("Tutti CUDA stream '
        'synchronization failed: %s", exc)\n'
        "                    streams_quiesced = False\n",
    )
    body = replace_once(
        body,
        "            with self._cpu_raw_writer_lock:\n",
        "            if streams_quiesced:\n"
        "                self._device_read_leases.clear()\n"
        "            with self._cpu_raw_writer_lock:\n",
    )
    source = source[:start] + body + source[end:]
    compile(source, "counted_lifetime_v379", "exec")
    return source


def install(root: Path) -> None:
    """Apply the opt-in lifetime patch during new-container startup.

    Args:
        root: Installed LMCache package directory.
    Raises:
        RuntimeError: Source does not match the expected public read methods.
    """
    path = root / "v1/gpu_connector/tutti_direct_loader.py"
    path.write_text(patch_source(path.read_text()))
    print("COUNTED_LIFETIME_V379_INSTALLED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    install(parser.parse_args().lmcache_root)
