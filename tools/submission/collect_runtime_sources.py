# SPDX-License-Identifier: Apache-2.0
"""Read Python source from stopped experiment containers without starting them."""

# Standard
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import zipfile

PACKAGE_ROOT = "/usr/local/lib/python3.12/dist-packages"


def read_package(container: str, package: str) -> dict[str, bytes]:
    """Stream only regular Python source from docker cp, ignoring binary assets.

    Args:
        container: Explicit container ID, including stopped containers.
        package: lmcache or vllm; arbitrary filesystem paths are not accepted.
    Returns:
        Package-relative source bytes. Raises on copy failure or unsafe members.
    Does not start, change, commit or delete the source container.
    """
    if package not in {"lmcache", "vllm"}:
        raise ValueError("Only named source packages are allowed")
    process = subprocess.Popen(
        ["docker", "cp", f"{container}:{PACKAGE_ROOT}/{package}", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    result = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if not member.isfile() or path.suffix != ".py":
                    continue
                if path.is_absolute() or ".." in path.parts or path.parts[0] != package:
                    raise ValueError("Unsafe source archive member")
                if member.size > 5 * 1024 * 1024:
                    raise ValueError(f"Unexpectedly large source file: {path}")
                result[str(path)] = archive.extractfile(member).read()
        if process.wait(timeout=30) != 0:
            raise RuntimeError(f"Cannot copy {package} from {container}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        if process.stdout is not None:
            process.stdout.close()
    if not result:
        raise ValueError("Empty source package")
    return result


def collect(containers: dict[str, str], output: Path) -> dict:
    """Archive exact source and source-only image deltas for named deployments.

    Creates an unstarted baseline container to read the common image. Preserves
    it for audit; no running or historical container is stopped or modified.
    Writes source/receipt files only under a new output directory.
    """
    output.mkdir(parents=True, exist_ok=False)
    identities = {}
    for model, container in containers.items():
        info = json.loads(subprocess.check_output(["docker", "inspect", container]))[0]
        if info["State"]["Running"]:
            raise ValueError("Capture requires a stopped source container")
        identities[model] = dict(
            container_id=info["Id"], image=info["Image"], created=info["Created"]
        )
    images = {item["image"] for item in identities.values()}
    if len(images) != 1:
        raise ValueError("Resolve different base images explicitly")
    base = subprocess.check_output(
        [
            "docker",
            "create",
            "--network",
            "none",
            "--read-only",
            "--entrypoint",
            "/bin/true",
            next(iter(images)),
        ],
        text=True,
    ).strip()
    baseline = {}
    for package in ("lmcache", "vllm"):
        baseline.update(read_package(base, package))
    receipt = dict(
        baseline_container=base, models=identities, source_hashes={}, deltas={}
    )
    target = output / "runtime_sources.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative, payload in baseline.items():
            bundle.writestr("base/" + relative, payload)
        for model, identity in identities.items():
            current = {}
            for package in ("lmcache", "vllm"):
                current.update(read_package(identity["container_id"], package))
            receipt["source_hashes"][model] = {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in current.items()
            }
            changed = []
            for relative, payload in current.items():
                if baseline.get(relative) != payload:
                    bundle.writestr(model + "/" + relative, payload)
                    changed.append(relative)
            receipt["deltas"][model] = dict(
                changed=sorted(changed), removed=sorted(set(baseline) - set(current))
            )
            print(
                json.dumps(dict(model=model, total=len(current), changed=len(changed))),
                flush=True,
            )
        bundle.writestr("PROVENANCE.json", json.dumps(receipt, indent=2))
    receipt["archive_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flash", required=True)
    parser.add_argument("--pro", required=True)
    parser.add_argument("--glm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(dict(flash=args.flash, pro=args.pro, glm=args.glm), args.output)
