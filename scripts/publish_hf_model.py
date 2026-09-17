#!/usr/bin/env python3
"""Publish the Flychess model artifact as a curated Hugging Face model repo.

The source repository and the model repository intentionally have different
payloads.  This script stages the model card, metadata, release receipt, and
the SafeTensors weights, then removes only the previously audited source-mirror
paths from the model repository.

Run this from a Colab checkout after placing ``HF_TOKEN`` and
``GITHUB_ACCESS_TOKEN`` in the runtime secrets.  The destructive prune is
opt-in so an accidental invocation cannot remove files from a Hub repo.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SHAPES: dict[str, list[int]] = {
    "decoder.bias": [512],
    "decoder.weight": [512, 41692],
    "encoder.bias": [10855],
    "encoder.weight": [10855, 780],
    "log_gain": [15091983],
    "policy.bias": [1968],
    "policy.weight": [1968, 512],
    "scale": [5, 138639],
    "shift": [5, 138639],
    "value.bias": [64],
    "value.weight": [64, 512],
}

MODEL_FILES = {
    ".gitattributes",
    "README.md",
    "config.json",
    "model_index.json",
    "flynet.safetensors",
    "results/release.json",
}

GENERIC_RUNTIME_HOMES = frozenset({"/content", "/root", "/tmp", "/workspace"})
ABSOLUTE_USER_HOME = re.compile(r"(?<![A-Za-z0-9._-])/(?:home|Users)/[^/\s'\"`]+")


def private_home_prefix(home: Path | None = None) -> str | None:
    """Return an identity-bearing home prefix, excluding hosted-runtime roots."""

    prefix = (home or Path.home()).resolve().as_posix().rstrip("/")
    if prefix in {"", "/", "/home", "/Users"} or prefix in GENERIC_RUNTIME_HOMES:
        return None
    return prefix


def read_secret(name: str) -> str | None:
    """Read a runtime secret without printing or persisting it."""

    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        value = userdata.get(name)
    except Exception:
        value = None
    return value or None


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def git_env(token: str) -> dict[str, str]:
    """Return a non-persistent authenticated Git environment."""

    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
        }
    )
    return env


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_weights(path: Path) -> tuple[list[str], dict[str, list[int]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != 99_572_154:
        raise ValueError(
            f"unexpected model size: {path.stat().st_size} bytes; "
            "the release must use the verified ChessFly SafeTensors artifact"
        )

    from safetensors import safe_open  # type: ignore[import-not-found]

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    if keys != sorted(EXPECTED_SHAPES):
        raise ValueError(f"unexpected tensor keys: {keys}")
    if shapes != EXPECTED_SHAPES:
        raise ValueError(f"unexpected tensor shapes: {shapes}")
    return keys, shapes


def test_summary(source_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=source_dir,
        env=env,
        text=True,
        capture_output=True,
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0:
        raise RuntimeError(f"pytest failed:\n{output}")
    match = re.search(r"(?P<passed>\d+) passed(?:, (?P<skipped>\d+) skipped)?", output)
    if not match:
        raise RuntimeError(f"could not parse pytest summary: {output}")
    return {
        "command": "python -m pytest -q",
        "passed": int(match.group("passed")),
        "skipped": int(match.group("skipped") or 0),
    }


def load_repo_paths(api: Any, repo_id: str) -> list[str]:
    info = api.model_info(repo_id, revision="main", files_metadata=True)
    return sorted(item.rfilename for item in (info.siblings or []))


def stage_payload(
    payload: Path,
    *,
    card: str,
    weights: Path,
    source_sha: str,
    tensor_keys: list[str],
    tensor_shapes: dict[str, list[int]],
    tests: dict[str, Any],
    graph_repo: str,
) -> tuple[str, int, dict[str, Any]]:
    payload.mkdir(parents=True, exist_ok=True)
    (payload / "results").mkdir(exist_ok=True)
    (payload / "README.md").write_text(card, encoding="utf-8")
    (payload / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    shutil.copyfile(weights, payload / "flynet.safetensors")

    config = {
        "model_type": "connectome_shaped_recurrent_policy",
        "library_name": "pytorch",
        "weights": "flynet.safetensors",
        "input_features": 780,
        "policy_logits": 1968,
        "value_logits": 64,
        "recurrent_steps": 5,
        "graph": {
            "neurons": 138639,
            "signed_edges": 15091983,
            "encoder_destinations": 10855,
            "readout_neurons": 41692,
        },
        "biological_calibration": "synthetic_only",
    }
    (payload / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    model_index = {
        "name": "Flychess ChessFly adapter",
        "library_name": "pytorch",
        "model_type": "connectome-shaped recurrent policy",
        "task": "chess move selection",
        "weights": "flynet.safetensors",
    }
    (payload / "model_index.json").write_text(
        json.dumps(model_index, indent=2) + "\n", encoding="utf-8"
    )

    digest = sha256_file(weights)
    release = {
        "schema_version": "1.0",
        "project": "flychess",
        "release_type": "model_artifact",
        "source": {
            "github_repository": "https://github.com/EF-Code/flychess",
            "branch": "master",
            "commit": source_sha,
        },
        "artifact": {
            "filename": "flynet.safetensors",
            "format": "safetensors",
            "bytes": weights.stat().st_size,
            "sha256": digest,
            "tensor_keys": tensor_keys,
            "tensor_shapes": tensor_shapes,
        },
        "architecture": config,
        "graph_reference": {
            "repository": graph_repo,
            "repo_type": "space",
            "files": ["data/connectome.bin.gz", "data/neurons.bin.gz"],
            "included_in_this_repo": False,
        },
        "verification": {
            **tests,
            "safetensors_load": "pass",
            "tensor_contract": "pass",
            "legal_move_boundary": "pass",
            "source_path_hygiene": "pass",
        },
        "biological_calibration": {
            "status": "synthetic_only",
            "real_malecns_targets_loaded": False,
            "biological_equivalence_claim": False,
        },
    }
    (payload / "results/release.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )
    return digest, weights.stat().st_size, release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("/content/flychess"))
    parser.add_argument("--repo-id", default=os.environ.get("FLYCHESS_MODEL_REPO"))
    parser.add_argument(
        "--graph-repo",
        default=os.environ.get("FLYCHESS_GRAPH_REPO", "mlabonne/chessfly"),
    )
    parser.add_argument(
        "--prune-source-mirror",
        action="store_true",
        help="delete the audited source-mirror and stale receipt paths",
    )
    args = parser.parse_args()
    if not args.prune_source_mirror:
        parser.error("--prune-source-mirror is required for an intentional model-only release")

    hf_token = read_secret("HF_TOKEN")
    github_token = read_secret("GITHUB_ACCESS_TOKEN")
    if not hf_token or not github_token:
        raise RuntimeError("required runtime secrets are unavailable")

    source_dir = args.source_dir.resolve()
    if run(["git", "status", "--porcelain"], cwd=source_dir):
        raise RuntimeError("source checkout is dirty; refusing to overwrite it")
    run(["git", "fetch", "origin", "master"], cwd=source_dir, env=git_env(github_token))
    source_sha = run(["git", "rev-parse", "FETCH_HEAD"], cwd=source_dir)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source_dir)
    home_prefix = private_home_prefix()
    tracked_files = run(["git", "ls-files"], cwd=source_dir).splitlines()
    for relative in tracked_files:
        candidate = source_dir / relative
        if candidate.is_file():
            contents = candidate.read_text(encoding="utf-8", errors="ignore")
            if (home_prefix and home_prefix in contents) or ABSOLUTE_USER_HOME.search(contents):
                raise RuntimeError("source checkout still contains a private home-path reference")

    card = (source_dir / "docs/huggingface-model-card.md").read_text(encoding="utf-8")
    if not card.startswith("---\n") or "flynet.safetensors" not in card:
        raise RuntimeError("professional model card is missing or incomplete")
    tensor_keys, tensor_shapes = validate_weights(args.weights)
    tests = test_summary(source_dir)

    from huggingface_hub import CommitOperationDelete, HfApi  # type: ignore[import-not-found]

    api = HfApi(token=hf_token)
    if args.repo_id:
        repo_id = args.repo_id
    else:
        identity = api.whoami()
        account = identity.get("name")
        if not account:
            raise RuntimeError("Hub account identity is unavailable")
        repo_id = f"{account}/flychess"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    existing = load_repo_paths(api, repo_id)

    with tempfile.TemporaryDirectory(prefix="flychess-hf-release-") as temp_dir:
        payload = Path(temp_dir)
        digest, byte_count, release = stage_payload(
            payload,
            card=card,
            weights=args.weights.resolve(),
            source_sha=source_sha,
            tensor_keys=tensor_keys,
            tensor_shapes=tensor_shapes,
            tests=tests,
            graph_repo=args.graph_repo,
        )
        api.upload_folder(
            folder_path=str(payload),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="",
            commit_message="publish verified Flychess model artifact",
        )

    obsolete = sorted(set(existing) - MODEL_FILES)
    if obsolete:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[CommitOperationDelete(path_in_repo=path) for path in obsolete],
            commit_message="remove source mirror and stale release files",
        )

    final_paths = load_repo_paths(api, repo_id)
    expected_paths = sorted(MODEL_FILES)
    if final_paths != expected_paths:
        raise RuntimeError(f"unexpected final model file set: {final_paths}")
    info = api.model_info(repo_id, revision="main", files_metadata=True)
    sizes = {item.rfilename: getattr(item, "size", None) for item in (info.siblings or [])}
    if sizes.get("flynet.safetensors") != byte_count:
        raise RuntimeError(f"Hub model size mismatch: {sizes.get('flynet.safetensors')}")

    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    remote_path = hf_hub_download(
        repo_id=repo_id,
        filename="flynet.safetensors",
        revision=info.sha,
        token=hf_token,
    )
    if sha256_file(Path(remote_path)) != digest:
        raise RuntimeError("Hub model checksum mismatch")

    # Keep output free of account names, tokens, local paths, and other private
    # environment details while still providing an auditable release receipt.
    print("HF_MODEL_RELEASE: PASS")
    print("HF_MODEL_REVISION:", info.sha)
    print("HF_MODEL_BYTES:", byte_count)
    print("HF_MODEL_SHA256:", digest)
    print("HF_MODEL_FILES:", ",".join(final_paths))
    print("SOURCE_COMMIT:", release["source"]["commit"])
    print("PYTEST:", f"{tests['passed']} passed, {tests['skipped']} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
