# SPDX-License-Identifier: Apache-2.0
"""Generate a complete private Tutti deployment plan without historical launchers."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path
import sys
import sysconfig

# Local
from model_profiles import environment
from warm_manifest import load_inputs


def prepare(config: dict, repo: Path, output: Path) -> dict:
    """Generate explicit build/server/warmup/verification commands on the target host.

    Args:
        config: Deployment settings, model and private storage/input paths.
        repo: Submission source root, separate from installed package paths.
        output: Fresh deployment state directory, never reused/overwritten.
    Returns:
        Complete cold-start plan. Raises on missing files, invalid model/backend
        or unsafe configuration. Never launches servers or changes SSD devices.
    Notes:
        This initial native startup path supports Tutti K and Tutti NP. The
        archived conventional SSD/DRAM experiment recipes remain separate until
        their public readiness adapter is integrated; do not silently skip it.
    """
    # Third Party
    import yaml

    repo, output = repo.resolve(), output.resolve()
    model, arm = config["model"], config.get("arm", "k_local")
    if arm not in {"k_local", "no_prefetch"}:
        raise ValueError("Native startup currently requires a Tutti path")
    flags = environment(repo / "config/serving_profiles.json", model, arm)
    checkpoint, inputs, storage_file = (
        Path(config[key]).resolve()
        for key in ("model_path", "warmup_inputs", "storage_config")
    )
    model_config = json.loads((checkpoint / "config.json").read_text())
    kind = model_config.get("model_type")
    if (model == "glm" and kind != "glm_moe_dsa") or (
        model != "glm" and kind != "deepseek_v4"
    ):
        raise ValueError("Checkpoint model_type differs from the selected model")
    maximum = config["max_model_len"]
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Explicit positive max_model_len required")
    manifest, _, _ = load_inputs(inputs, maximum)
    storage = yaml.safe_load(storage_file.read_text())
    # File-backed config is authoritative; environment flags alone do not
    # populate SSDTPShardedPrefetchConfig.from_engine_config().
    storage.update(
        ssd_tp_sharded_prefetch=True,
        ssd_tp_shard_indexer=False,
        ssd_tp_shard_csa=True,
        ssd_tp_csa_replica_verified=True,
        ssd_tp_indexer_cp_verified=False,
        ssd_tp_cp_size=8,
        ssd_tp_cp_interleave=64,
        ssd_tp_staging_slot_bytes=268435456,
        ssd_tp_staging_slots=3,
        ssd_tp_early_lookahead=2,
    )
    extra = storage.get("extra_config", {})
    if (
        storage.get("chunk_size") != 256
        or storage.get("local_cpu") is not False
        or storage.get("internal_api_server_enabled") is not True
        or storage.get("internal_api_server_host") != "127.0.0.1"
        or extra.get("kv_object_store_tutti_raw_enable") is not True
        or storage.get("local_disk_path_sharding") != "by_gpu"
    ):
        raise ValueError(
            "Storage must declare a dedicated raw Tutti TP8 cache and private API"
        )
    disks = str(storage["local_disk"]).split(",")
    regions = str(extra["kv_object_store_tutti_raw_region_path"]).split(",")
    indexer_regions = config["indexer_raw_regions"]
    indexer_directories = config["indexer_directories"]
    for paths in (disks, regions, indexer_regions, indexer_directories):
        if (
            len(paths) != 8
            or len(set(paths)) != 8
            or any(not Path(p).is_absolute() for p in paths)
        ):
            raise ValueError("Eight distinct absolute storage paths required")
    if config.get("dedicated_cache") is not True:
        raise ValueError(
            "Explicit dedicated_cache=true required before startup cache clearing"
        )
    kv_bytes, utilization = (
        config["kv_cache_memory_bytes"],
        config["gpu_memory_utilization"],
    )
    if type(kv_bytes) is not int or kv_bytes <= 0 or not 0 < utilization < 1:
        raise ValueError("Explicit KV budget and GPU utilization required")
    port = config.get("port", 8000)
    api_port = storage["internal_api_server_port_start"]
    if not isinstance(api_port, int) or not 1 <= api_port <= 65527:
        raise ValueError("Invalid worker API port range")
    endpoint = f"http://127.0.0.1:{port}"
    ranks = [f"http://127.0.0.1:{api_port + 1 + rank}" for rank in range(8)]
    if endpoint in ranks or port == api_port or not 1 <= port <= 65535:
        raise ValueError("Inference port conflicts with internal rank APIs")
    tool = repo / "tools/submission"
    site = Path(sysconfig.get_paths()["purelib"])
    installed_lmcache = site / "lmcache"
    installed_vllm = site / "vllm"
    native_source = repo / "experiments/native/device_count_v137"
    native_receipt = output / "COUNTED_BUILD.json"
    flags.update(
        LMCACHE_CONFIG_FILE=str(output / "lmcache.yaml"),
        LMCACHE_INDEXER_SSD_DIR=",".join(indexer_directories),
        LMCACHE_INDEXER_TUTTI_RAW_REGION_PATH=",".join(indexer_regions),
        LMCACHE_INDEXER_MAX_SEQ_LEN=str(maximum),
        LMCACHE_LOOKUP_SERVER_WORKER_IDS="0,1,2,3,4,5,6,7",
        JANUS_COUNTED_BUILD_RECEIPT=str(native_receipt),
        JANUS_COUNTED_SOURCE_ROOT=str(native_source),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
        TORCH_CUDA_ARCH_LIST="9.0",
        MAX_JOBS="2",
        BUILD_MOONCAKE="0",
        PYTHONHASHSEED="0",
        LMCACHE_TUTTI_HOST_MOUNT_HANDOFF=(
            "1" if config.get("host_mount_handoff", False) else "0"
        ),
    )
    worker_env = output / "worker_env.json"
    python = sys.executable
    commands = dict(
        build=[
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                str(repo),
            ],
            [
                python,
                str(tool / "apply_overlay.py"),
                "--source",
                str(repo),
                "--model",
                model,
                "--lmcache-root",
                str(installed_lmcache),
                "--vllm-root",
                str(installed_vllm),
            ],
            [
                python,
                str(tool / "build_native.py"),
                "--source",
                str(native_source),
                "--cache",
                str(output / "native-cache"),
                "--arch",
                "9.0",
                "--jobs",
                "2",
                "--receipt",
                str(native_receipt),
            ],
            [python, str(tool / "verify_runtime.py")],
        ],
        server=[
            python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(checkpoint),
            "--tokenizer",
            str(Path(config.get("tokenizer_path", checkpoint)).resolve()),
            "--served-model-name",
            manifest["model"],
            "--generation-config",
            "vllm",
            "--tensor-parallel-size",
            "8",
            "--enable-expert-parallel",
            "--enable-ep-weight-filter",
            "--all2all-backend",
            "allgather_reducescatter",
            "--enforce-eager",
            "--kv-cache-dtype",
            "fp8",
            "--max-model-len",
            str(maximum),
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            str(config.get("max_num_batched_tokens", 8192)),
            "--enable-chunked-prefill",
            "--gpu-memory-utilization",
            str(utilization),
            "--kv-cache-memory-bytes",
            str(kv_bytes),
            "--no-scheduler-reserve-full-isl",
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}',
            "--no-disable-hybrid-kv-cache-manager",
            "--no-enable-prefix-caching",
            "--trust-remote-code",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        warmup=[
            python,
            str(tool / "warm_manifest.py"),
            "--inputs",
            str(inputs),
            "--endpoint",
            endpoint,
            "--rank-endpoints",
            *ranks,
            "--worker-env",
            str(worker_env),
            "--max-model-len",
            str(maximum),
            "--dedicated-cache",
        ],
        verify=[
            python,
            str(tool / "verify_ready.py"),
            "--endpoint",
            endpoint,
            "--rank-endpoints",
            *ranks,
            "--worker-env",
            str(worker_env),
            "--model-name",
            manifest["model"],
        ],
    )
    plan = dict(
        identity=dict(
            model_family=model,
            model_revision=hashlib.sha256(
                (checkpoint / "config.json").read_bytes()
            ).hexdigest(),
            gpu_arch="sm_90",
            tp_size=8,
            precision="fp8 KV; native model indexer dtype",
            shape_policy=hashlib.sha256(
                (inputs / "manifest.json").read_bytes()
            ).hexdigest(),
        ),
        env=flags,
        isolate_lmcache_env=True,
        health_url=endpoint + "/health",
        health_timeout_s=1800,
        command_timeout_s=7200,
        **commands,
    )
    output.mkdir(parents=True, exist_ok=False)
    extra.update(
        tutti_startup_warmup_delay_sec=0.1,
        tutti_startup_health_port=port,
        tutti_startup_health_poll_interval_sec=1.0,
        tutti_startup_health_poll_timeout_sec=1800,
    )
    (output / "lmcache.yaml").write_text(yaml.safe_dump(storage), encoding="utf-8")
    worker_env.write_text(json.dumps(flags, indent=2))
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.deployment.read_text()), args.source, args.output)
    print(json.dumps(dict(plan=str(args.output / "plan.json"), started=False)))
