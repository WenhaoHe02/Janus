# SPDX-License-Identifier: Apache-2.0
"""Install opt-in experiment telemetry before the worker processes start.

This runs through normal container startup, not the restricted script API.
Only the dedicated measurement startup invokes it. Prediction math is unchanged.
"""

# Standard
import argparse
from pathlib import Path


def install(root: Path) -> None:
    """Add rank-local exact-set files and connect the decode-only observer."""
    predictor = root / "v1/glm_dsa_predictive_prefetch.py"
    text = predictor.read_text()
    if "GLM_DSA_EXACT_SETS_FILE_V317" not in text:
        if "GLM_DSA_EXACT_SETS_V307" in text:
            raise RuntimeError("exact-set upgrade requires the clean startup overlay")
        needle = (
            "        predicted = predicted.to("
            "device=actual.device, dtype=torch.int64)\n"
        )
        if text.count(needle) != 1:
            raise RuntimeError("unrecognized GLM block-agreement implementation")
        insertion = """        exact_record = {
                "rank": int(torch.distributed.get_rank()),
                "request_id": event.request_id,
                "source": int(event.schedule.source_layer),
                "target": int(event.schedule.target_layer),
                "block_size": int(self._compressed_block_size),
                "predicted": predicted.detach().cpu().to(dtype=torch.int64).tolist(),
                "actual": actual.detach().cpu().to(dtype=torch.int64).tolist(),
        }
        exact_request = str(event.request_id)
        if not exact_request or not all(
            c.isascii() and (c.isalnum() or c in "-_") for c in exact_request
        ):
            raise ValueError("unsafe exact-set request identifier")
        exact_directory = os.path.join(
            os.getenv("LMCACHE_GLM_EXACT_SETS_DIR", "/tmp/glm_exact_sets_v317"),
            exact_request,
        )
        os.makedirs(exact_directory, exist_ok=True)
        exact_path = os.path.join(
            exact_directory, "rank%d.jsonl" % exact_record["rank"]
        )
        with self._lock:
            with open(exact_path, "a", encoding="utf-8") as exact_file:
                exact_file.write(json.dumps(exact_record, separators=(",", ":")) + "\\n")
        logger.info("GLM_DSA_EXACT_SETS_FILE_V317 %s", json.dumps({
            "rank": exact_record["rank"], "request_id": event.request_id,
            "target": int(event.schedule.target_layer), "path": exact_path,
        }, separators=(",", ":")))
"""
        if "import json\n" not in text:
            text = text.replace("import math\n", "import json\nimport math\n", 1)
        text = text.replace(needle, insertion + needle, 1)
        compile(text, str(predictor), "exec")
        predictor.write_text(text)

    text = predictor.read_text()
    if "target=L%d rank=%d " not in text:
        old_format = '"GLM DSA physical block coverage request=%s target=L%d "'
        old_args = (
            "            event.request_id,\n"
            "            event.schedule.target_layer,\n"
            "            int(predicted.numel()),\n"
        )
        if text.count(old_format) != 1 or text.count(old_args) != 1:
            raise RuntimeError("unrecognized physical coverage counter logging")
        text = text.replace(
            old_format,
            '"GLM DSA physical block coverage request=%s target=L%d rank=%d "',
            1,
        )
        text = text.replace(
            old_args,
            "            event.request_id,\n"
            "            event.schedule.target_layer,\n"
            "            int(torch.distributed.get_rank()),\n"
            "            int(predicted.numel()),\n",
            1,
        )
    if "GLM_DSA_DECODE_COUNTS_V307" not in text:
        changes = [
            (
                "        self._decode_missing_blocks: dict[int, set[int]] "
                "= defaultdict(set)\n",
                "        self._decode_missing_blocks: dict[int, set[int]] "
                "= defaultdict(set)\n"
                "        self._decode_observation_totals: dict[int, list[int]] = {}\n",
            ),
            (
                "            selected, missing = query(int(layer_id), true_topk)\n"
                "            with self._lock:\n",
                "            selected, missing = query(int(layer_id), true_topk)\n"
                "            with self._lock:\n"
                "                totals = self._decode_observation_totals.setdefault(\n"
                "                    int(layer_id), [0, 0, 0])\n"
                "                totals[0] += 1\n"
                "                totals[1] += int(selected.numel())\n"
                "                totals[2] += int(missing.numel())\n",
            ),
            (
                "        if selected:\n            logger.info(\n"
                '                "GLM_DSA_DECODE_COVERAGE',
                "        if selected:\n"
                '            logger.info("GLM_DSA_DECODE_COUNTS_V307 %s", '
                "json.dumps({\n"
                "                'request_id': request_id,\n"
                "                'rank': int(torch.distributed.get_rank()),\n"
                "                'selected_unique_blocks': selected,\n"
                "                'missing_unique_blocks': missing,\n"
                "                'per_layer_steps_selected_missing':\n"
                "                    self._decode_observation_totals,\n"
                "                'is_actual_io': False,\n"
                "            }, separators=(',', ':')))\n"
                '            logger.info(\n                "GLM_DSA_DECODE_COVERAGE',
            ),
            (
                "            self._decode_missing_blocks.clear()\n",
                "            self._decode_missing_blocks.clear()\n"
                "            self._decode_observation_totals.clear()\n",
            ),
        ]
        for anchor, replacement in changes:
            if text.count(anchor) != 1:
                raise RuntimeError(
                    f"unrecognized decode observation anchor: {anchor[:70]}"
                )
            text = text.replace(anchor, replacement, 1)
        compile(text, str(predictor), "exec")
        predictor.write_text(text)

    adapter = root / "integration/vllm/vllm_v1_adapter.py"
    text = adapter.read_text()
    if "def glm_profile_decode_authoritative_v307(" not in text:
        needle = "def _glm_dsa_wait_for_consumer("
        if text.count(needle) != 1:
            raise RuntimeError("unrecognized GLM consumer callback implementation")
        helper = '''def glm_profile_decode_authoritative_v307(
    target_layer: int, true_topk: torch.Tensor
) -> None:
    """Send decode selections to the existing opt-in coverage observer."""
    sink = _GLM_DSA_PHYSICAL_PREFETCH_SINK
    if sink is not None:
        sink.profile_decode_authoritative(target_layer, true_topk)


'''
        text = text.replace(needle, helper + needle, 1)
        anchor = "            consumer_waiter=_glm_dsa_wait_for_consumer,\n"
        if text.count(anchor) != 1:
            raise RuntimeError("unrecognized GLM hook construction")
        text = text.replace(
            anchor,
            anchor + "            disabled_authoritative_observer="
            "glm_profile_decode_authoritative_v307,\n",
            1,
        )
        compile(text, str(adapter), "exec")
        adapter.write_text(text)
    from install_glm_phase_v320 import install as install_phase

    install_phase(root)
    print("GLM_V307_MEASUREMENT_INSTALLED", str(root), flush=True)


def main() -> None:
    """Install telemetry into the explicit container package directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmcache-root", type=Path, required=True)
    args = parser.parse_args()
    install(args.lmcache_root)


if __name__ == "__main__":
    main()
