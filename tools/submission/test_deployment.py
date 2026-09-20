# SPDX-License-Identifier: Apache-2.0
"""Check the complete versionless deployment plan without launching a model."""

# Standard
from array import array
import json
from pathlib import Path
import tempfile
import unittest

# Local
from model_profiles import environment
from prepare_deployment import prepare
from start import validate_model_plan


def fixture(root: Path, model: str) -> tuple[dict, Path]:
    """Create synthetic test-only configuration files, not benchmark content."""
    repo = root / "repo"
    (repo / "config").mkdir(parents=True)
    flags = dict(LMCACHE_PROXY_INDEXER_CP="0", LMCACHE_TRUE_INDEXER_CP="0")
    profiles = {
        family: {arm: flags for arm in ("k_local", "no_prefetch", "ssd_only", "dram")}
        for family in ("dsv4", "glm")
    }
    (repo / "config/serving_profiles.json").write_text(json.dumps(profiles))
    checkpoint, inputs = root / "model", root / "inputs"
    checkpoint.mkdir()
    inputs.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(dict(model_type="glm_moe_dsa" if model == "glm" else "deepseek_v4"))
    )
    (inputs / "tokens.i32le").write_bytes(array("i", range(280)).tobytes())
    row = dict(
        source_request_id="a",
        token_offset=0,
        prompt_tokens=276,
        reusable_input_prefix_tokens=257,
    )
    (inputs / "manifest.json").write_text(
        json.dumps(
            dict(
                model="test-model",
                selected_count=1,
                chains=[dict(first_timed_index=0, rounds=[row])],
            )
        )
    )
    storage = root / "storage.yaml"
    paths = [str(root / f"cache/rank{rank}") for rank in range(8)]
    regions = [str(root / f"raw/rank{rank}") for rank in range(8)]
    storage.write_text(
        json.dumps(
            dict(
                chunk_size=256,
                local_cpu=False,
                internal_api_server_enabled=True,
                internal_api_server_host="127.0.0.1",
                internal_api_server_port_start=6999,
                local_disk=",".join(paths),
                local_disk_path_sharding="by_gpu",
                extra_config=dict(
                    kv_object_store_tutti_raw_enable=True,
                    kv_object_store_tutti_raw_region_path=",".join(regions),
                ),
            )
        )
    )
    config = dict(
        model=model,
        arm="k_local",
        model_path=str(checkpoint),
        warmup_inputs=str(inputs),
        storage_config=str(storage),
        max_model_len=512,
        indexer_raw_regions=[
            str(root / f"indexer_raw/rank{rank}") for rank in range(8)
        ],
        indexer_directories=[str(root / f"indexer/rank{rank}") for rank in range(8)],
        dedicated_cache=True,
        kv_cache_memory_bytes=1024,
        gpu_memory_utilization=0.5,
    )
    return config, repo


class DeploymentTests(unittest.TestCase):
    """Generated plans keep all build/warmup steps before readiness."""

    def test_three_models(self) -> None:
        """All three models emit self-contained versionless commands."""
        for model in ("flash", "pro", "glm"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config, repo = fixture(root, model)
                plan = prepare(config, repo, root / "deployment")
                validate_model_plan(root / "deployment/plan.json", model)
                self.assertEqual(
                    plan["server"][plan["server"].index("--host") + 1], "127.0.0.1"
                )
                self.assertIn("--dedicated-cache", plan["warmup"])
                self.assertNotIn("--legacy", plan["warmup"])
                self.assertTrue(plan["isolate_lmcache_env"])
                self.assertEqual(plan["env"]["PYTHONHASHSEED"], "0")
                self.assertTrue(
                    all(
                        "_v" not in Path(command[1]).name
                        for command in plan["build"][1:]
                    )
                )
                self.assertTrue((root / "deployment/worker_env.json").is_file())
                import yaml

                emitted = yaml.safe_load((root / "deployment/lmcache.yaml").read_text())
                self.assertTrue(emitted["ssd_tp_sharded_prefetch"])
                self.assertTrue(emitted["ssd_tp_csa_replica_verified"])
                self.assertFalse(emitted["ssd_tp_shard_indexer"])

    def test_wrong_model_or_shared_cache(self) -> None:
        """Model identity and dedicated cache ownership cannot be implicit."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, repo = fixture(root, "flash")
            for change in (dict(model="glm"), dict(dedicated_cache=False)):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    prepare(dict(config, **change), repo, root / "not_created")
            self.assertFalse((root / "not_created").exists())

    def test_profile_has_measured_owner_controls(self) -> None:
        """GLM device-owner and DS GPU metadata are explicit rather than defaults."""
        with tempfile.TemporaryDirectory() as folder:
            _, repo = fixture(Path(folder), "flash")
            profile = repo / "config/serving_profiles.json"
            self.assertEqual(
                environment(profile, "glm", "k_local")[
                    "LMCACHE_GLM_DSA_DEVICE_OWNER_READ"
                ],
                "1",
            )
            self.assertEqual(
                environment(profile, "glm", "no_prefetch")[
                    "LMCACHE_GLM_DSA_DEVICE_OWNER_READ"
                ],
                "0",
            )
            self.assertEqual(
                environment(profile, "flash", "k_local")[
                    "LMCACHE_CSA_OWNER_GPU_METADATA"
                ],
                "1",
            )


if __name__ == "__main__":
    unittest.main()
