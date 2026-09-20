# SPDX-License-Identifier: Apache-2.0
"""Install opt-in raw-write counters and phase markers before worker startup.

Counts successful O_DIRECT writes including sector padding. Never changes
write admission, block geometry, read selection, or CUDA stream ordering.
"""

# Standard
import argparse
from pathlib import Path


def replace_once(text: str, anchor: str, replacement: str) -> str:
    """Replace a verified source anchor, rejecting an unknown implementation."""
    if text.count(anchor) != 1:
        raise RuntimeError(f"unrecognized I/O telemetry anchor: {anchor[:70]}")
    return text.replace(anchor, replacement, 1)


def install(root: Path) -> None:
    """Add counters to the explicit installed LMCache package directory.

    Args:
        root: Installed LMCache package; frozen source overlays are untouched.
    Raises:
        RuntimeError: If the deployed implementation no longer matches.
    """
    (root / "v1/io_profile_records.py").write_bytes(
        Path("/tmp/io_profile_records_v318.py").read_bytes()
    )
    (root / "integration/vllm/glm_layer_nvtx.py").write_bytes(
        Path("/tmp/glm_layer_nvtx_v319.py").read_bytes()
    )
    hooks = root / "integration/vllm/glm_dsa_vllm_0202.py"
    hook_text = hooks.read_text()
    if "GLMLayerNvtxObserver" not in hook_text:
        anchor = "        self._forward_enabled = forward_enabled\n"
        hook_text = replace_once(
            hook_text,
            anchor,
            anchor
            + """        self._layer_profile_observer = None
        if os.getenv("LMCACHE_GLM_LAYER_NVTX", "0") == "1":
            from lmcache.integration.vllm.glm_layer_nvtx import GLMLayerNvtxObserver
            from lmcache.integration.vllm.glm_forward_phase import is_prefill
            self._layer_profile_observer = GLMLayerNvtxObserver(is_prefill)
""",
        )
        anchor = "        self._apply_indexshare()\n"
        hook_text = replace_once(
            hook_text,
            anchor,
            anchor
            + """        if self._layer_profile_observer is not None:
            for layer_id, decoder in self._decoder_layers.items():
                self._layer_profile_observer.bind_decoder(decoder, layer_id)
""",
        )
        anchor = "        for wrapper, indexer in self._shared_indexers:\n"
        hook_text = replace_once(
            hook_text,
            anchor,
            """        if self._layer_profile_observer is not None:
            self._layer_profile_observer.close()
"""
            + anchor,
        )
        compile(hook_text, str(hooks), "exec")
        hooks.write_text(hook_text)
    if "def _predict_activation_profile_body(" not in hook_text:
        anchor = "    def _predict_activation(\n"
        wrapper = '''    def _predict_activation(
        self,
        schedule: GLMDSAPredictionSchedule,
        activation: torch.Tensor,
        positions: torch.Tensor,
        phase_events: tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event]
        | None = None,
    ) -> torch.Tensor:
        """Mark only proxy GPU launches, on synchronous or asynchronous paths."""
        profile = os.getenv("LMCACHE_GLM_LAYER_NVTX", "0") == "1"
        if profile:
            torch.cuda.nvtx.range_push(
                "event=glm_proxy_indexer_launch|kind=glm_dsa|phase=prefill"
                f"|layer={schedule.source_layer}|target={schedule.target_layer}"
            )
        try:
            return self._predict_activation_profile_body(
                schedule, activation, positions, phase_events=phase_events
            )
        finally:
            if profile:
                torch.cuda.nvtx.range_pop()

    def _predict_activation_profile_body(
'''
        hook_text = replace_once(hook_text, anchor, wrapper)
        compile(hook_text, str(hooks), "exec")
        hooks.write_text(hook_text)
    writer = root / "v1/storage_backend/cpu_raw_writer.py"
    text = writer.read_text()
    if "TUTTI_CPU_RAW_WRITE_FILE_V318" not in text:
        text = replace_once(text, "import mmap\n", "import json\nimport mmap\n")
        anchor = "_LBA_BYTES = 512\n"
        text = replace_once(
            text,
            anchor,
            "# First Party\nfrom lmcache.logging import init_logger\n\n"
            "logger = init_logger(__name__)\n\n" + anchor,
        )
        anchor = "        wait_started_s = time.perf_counter()\n"
        text = replace_once(
            text, anchor, "        profile_start_ns = time.time_ns()\n" + anchor
        )
        anchor = (
            "        return normalized, (time.perf_counter() - started_s) * 1000.0\n"
        )
        replacement = """        elapsed_ms = (time.perf_counter() - started_s) * 1000.0
        if os.getenv("LMCACHE_GLM_IO_VOLUME_PROFILE", "0") == "1":
            from lmcache.v1.io_profile_records import append_profile_record
            record = {
                "rank": int(os.getenv("LOCAL_RANK", os.getenv("RANK", "-1"))),
                "pid": os.getpid(), "device": self.device_path,
                "start_host_ns": profile_start_ns,
                "end_host_ns": time.time_ns(),
                "payload_bytes": payload_nbytes,
                "completed_bytes": sum(extent[2] * _LBA_BYTES for extent in normalized),
                "extents": normalized, "elapsed_ms": elapsed_ms,
                "measurement": "successful O_DIRECT bytes including alignment padding",
            }
            reference = append_profile_record("cpu_raw_write", record)
            logger.info("TUTTI_CPU_RAW_WRITE_FILE_V318 %s",
                        json.dumps(reference, separators=(",", ":")))
        return normalized, elapsed_ms
"""
        text = replace_once(text, anchor, replacement)
        compile(text, str(writer), "exec")
        writer.write_text(text)

    loader = root / "v1/gpu_connector/tutti_direct_loader.py"
    text = loader.read_text()
    if '"host_completion_ns": time.time_ns(),' not in text:
        anchor = '            "schema": "tutti_device_batch_v1",\n'
        text = replace_once(
            text, anchor, anchor + '            "host_completion_ns": time.time_ns(),\n'
        )
        compile(text, str(loader), "exec")
        loader.write_text(text)

    text = loader.read_text()
    if "TUTTI_DEVICE_BATCH_FILE_V318" not in text:
        anchor = """        logger.info(
            "TUTTI_DEVICE_BATCH %s",
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
        )
"""
        replacement = """        from lmcache.v1.io_profile_records import (
            append_profile_record,
        )
        reference = append_profile_record("tutti_device", payload)
        logger.info("TUTTI_DEVICE_BATCH_FILE_V318 %s",
                    json.dumps(reference, separators=(",", ":")))
"""
        text = replace_once(text, anchor, replacement)
        compile(text, str(loader), "exec")
        loader.write_text(text)

    predictor = root / "v1/glm_dsa_predictive_prefetch.py"
    text = predictor.read_text()
    if "GLM_FIRST_DECODE_FORWARD_V310" not in text:
        anchor = (
            "        self._decode_missing_blocks: dict[int, set[int]] "
            "= defaultdict(set)\n"
        )
        text = replace_once(
            text,
            anchor,
            anchor + "        self._io_decode_marker_request: str | None = None\n",
        )
        anchor = (
            "        if not _decode_coverage_profile_enabled():\n            return\n"
        )
        replacement = (
            """        if os.getenv("LMCACHE_GLM_IO_VOLUME_PROFILE", "0") == "1":
            request_id = str(self._attention_kv_manager.active_request_id or "")
            if request_id and request_id != self._io_decode_marker_request:
                self._io_decode_marker_request = request_id
                logger.info("GLM_FIRST_DECODE_FORWARD_V310 %s", json.dumps({
                    "request_id": request_id, "target_layer": int(target_layer),
                    "rank": int(torch.distributed.get_rank()),
                    "pid": os.getpid(),
                    "host_ns": time.time_ns(),
                    "boundary": "first scheduler-confirmed decode authoritative observation",
                }, separators=(",", ":")))
"""
            + anchor
        )
        text = replace_once(text, anchor, replacement)
        compile(text, str(predictor), "exec")
        predictor.write_text(text)
    print("GLM_IO_MEASUREMENT_V310_INSTALLED", str(root), flush=True)


def main() -> None:
    """Install after the standard exact-set/observer startup patch."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", required=True, type=Path)
    args = parser.parse_args()
    install(args.lmcache_root)


if __name__ == "__main__":
    main()
