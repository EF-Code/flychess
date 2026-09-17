#!/usr/bin/env python3
"""Publish a verified, model-only FlyNet release to Hugging Face.

The release directory is produced by ``train_flynet.py``.  This publisher
uploads weights, the generated graph, configuration, training evidence, and a
release receipt; it does not mirror the Python source tree.  Use ``--prune``
only when the target repository is intentionally being converted to this
model-only payload.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from flychess.flynet import FLYNET_FEATURES, FLYNET_POLICY_ACTIONS, load_graph
from flychess.flynet_training import load_dataset, sha256_file


BASE_MODEL_FILES = {
    ".gitattributes",
    "README.md",
    "config.json",
    "flynet-graph.json",
    "flynet-graph.npz",
    "flynet.safetensors",
    "model_index.json",
    "results/release.json",
    "results/training.json",
}
GENERIC_RUNTIME_HOMES = frozenset({"/content", "/root", "/tmp", "/workspace"})
ABSOLUTE_USER_HOME = re.compile(r"(?<![A-Za-z0-9._-])/(?:home|Users)/[^/\s'\"`]+")


def private_home_prefix(home: Path | None = None) -> str | None:
    """Return an identity-bearing home prefix, excluding hosted roots."""

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
    result = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def git_env(token: str) -> dict[str, str]:
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


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact must be an object: {path}")
    return value


def validate_release(release_dir: Path) -> tuple[dict[str, Any], dict[str, list[int]], dict[str, Any]]:
    required = {
        "flynet.safetensors",
        "flynet-graph.npz",
        "flynet-graph.json",
        "flynet-config.json",
        "results/training.json",
    }
    missing = sorted(relative for relative in required if not (release_dir / relative).is_file())
    if missing:
        raise RuntimeError(f"FlyNet release is missing files: {missing}")
    config = read_object(release_dir / "flynet-config.json")
    if config.get("schema_version") != "flynet.config/v1":
        raise RuntimeError("release config is not a FlyNet v1 config")
    if config.get("model_type") != "flynet.signed_recurrent_graph":
        raise RuntimeError("release config is not an independent FlyNet model")
    if config.get("pretrained_weights_used") is not False:
        raise RuntimeError("release does not prove random initialization")
    if config.get("features") != FLYNET_FEATURES or config.get("actions") != FLYNET_POLICY_ACTIONS:
        raise RuntimeError("release config does not match the FlyNet runtime contract")
    graph = load_graph(release_dir / "flynet-graph.npz", release_dir / "flynet-graph.json")
    configured_graph = config.get("graph")
    if configured_graph != graph.summary():
        raise RuntimeError("release config graph summary does not match graph artifact")
    from safetensors import safe_open  # type: ignore[import-not-found]

    with safe_open(str(release_dir / "flynet.safetensors"), framework="pt", device="cpu") as handle:
        tensor_shapes = {key: list(handle.get_slice(key).get_shape()) for key in sorted(handle.keys())}
    if not tensor_shapes:
        raise RuntimeError("FlyNet weights contain no tensors")
    graph_summary = config["graph"]
    width = int(config["readout_width"])
    expected_shapes = {
        "edge_gain": [graph_summary["edges"]],
        "edge_sign": [graph_summary["edges"]],
        "encoder.0.bias": [graph_summary["inputs"]],
        "encoder.0.weight": [graph_summary["inputs"], FLYNET_FEATURES],
        "encoder.1.bias": [graph_summary["inputs"]],
        "encoder.1.weight": [graph_summary["inputs"]],
        "node_bias": [graph_summary["nodes"]],
        "node_scale": [int(config["steps"]), graph_summary["nodes"]],
        "policy_head.bias": [FLYNET_POLICY_ACTIONS],
        "policy_head.weight": [FLYNET_POLICY_ACTIONS, width],
        "readout.0.bias": [width],
        "readout.0.weight": [width, graph_summary["readout"]],
        "readout.1.bias": [width],
        "readout.1.weight": [width],
        "value_head.bias": [1],
        "value_head.weight": [1, width],
    }
    if tensor_shapes != expected_shapes:
        raise RuntimeError(f"FlyNet tensor contract mismatch: {tensor_shapes}")
    training = read_object(release_dir / "results/training.json")
    if training.get("schema_version") != "flynet.training/v1":
        raise RuntimeError("release training receipt is not a FlyNet v1 receipt")
    return config, tensor_shapes, training


def stage_payload(
    payload: Path,
    *,
    release_dir: Path,
    card: str,
    source_sha: str,
    tests: dict[str, Any],
    config: dict[str, Any],
    tensor_shapes: dict[str, list[int]],
    training: dict[str, Any],
    dataset_path: Path | None,
) -> tuple[set[str], dict[str, Any]]:
    payload.mkdir(parents=True, exist_ok=True)
    (payload / "results").mkdir(exist_ok=True)
    (payload / "README.md").write_text(card, encoding="utf-8")
    (payload / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n*.npz filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    copied = [
        "flynet.safetensors",
        "flynet-graph.npz",
        "flynet-graph.json",
        "flynet-config.json",
        "results/training.json",
    ]
    for relative in copied:
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release_dir / relative, destination)

    if dataset_path is not None:
        dataset_path = dataset_path.resolve()
        dataset = load_dataset(dataset_path)
        shutil.copyfile(dataset_path, payload / "flynet-dataset.npz")
        dataset_metadata = dict(dataset.metadata)
        dataset_metadata.update(
            {
                "schema_version": "flynet.dataset/v1",
                "samples": dataset.sample_count,
                "features": FLYNET_FEATURES,
                "actions": FLYNET_POLICY_ACTIONS,
                "file": "flynet-dataset.npz",
                "sha256": sha256_file(dataset_path),
            }
        )
        (payload / "flynet-dataset.json").write_text(
            json.dumps(dataset_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    model_index = {
        "name": "Flychess FlyNet",
        "library_name": "pytorch",
        "model_type": "signed recurrent sparse graph policy",
        "task": "chess move selection",
        "tags": ["chess", "sparse-neural-network", "supervised-learning", "flychess"],
        "weights": "flynet.safetensors",
    }
    (payload / "model_index.json").write_text(
        json.dumps(model_index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact_names = [
        "flynet.safetensors",
        "flynet-graph.npz",
        "flynet-graph.json",
        "flynet-config.json",
        "results/training.json",
    ]
    if dataset_path is not None:
        artifact_names.extend(["flynet-dataset.npz", "flynet-dataset.json"])
    files = {
        relative: {
            "bytes": (payload / relative).stat().st_size,
            "sha256": sha256_file(payload / relative),
        }
        for relative in artifact_names
    }
    release = {
        "schema_version": "flynet.release/v1",
        "project": "flychess",
        "release_type": "from_scratch_model_artifact",
        "source": {
            "github_repository": "https://github.com/EF-Code/flychess",
            "branch": "master",
            "commit": source_sha,
        },
        "provenance": {
            "pretrained_weights_used": False,
            "graph_generated_from_seed": config["graph"]["seed"],
            "training_initialization": config["initialization"],
            "teacher_labels": config["teacher_labels"],
        },
        "architecture": config,
        "tensor_shapes": tensor_shapes,
        "artifacts": files,
        "training": training,
        "verification": {
            **tests,
            "safetensors_load": "pass",
            "graph_contract": "pass",
            "config_contract": "pass",
            "source_path_hygiene": "pass",
            "legal_move_boundary": "runtime-enforced",
        },
        "scientific_boundary": {
            "biological_connectome_reconstruction": False,
            "biological_equivalence_claim": False,
            "teacher_distillation": True,
        },
    }
    (payload / "results/release.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected = {
        ".gitattributes",
        "README.md",
        "flynet-config.json",
        "flynet-graph.json",
        "flynet-graph.npz",
        "flynet.safetensors",
        "model_index.json",
        "results/release.json",
        "results/training.json",
    }
    if dataset_path is not None:
        expected.update({"flynet-dataset.json", "flynet-dataset.npz"})
    return expected, release


def load_repo_paths(api: Any, repo_id: str) -> list[str]:
    info = api.model_info(repo_id, revision="main", files_metadata=True)
    return sorted(item.rfilename for item in (info.siblings or []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("/content/flychess"))
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--repo-id", default=os.environ.get("FLYCHESS_MODEL_REPO"))
    parser.add_argument("--prune", action="store_true", help="delete files outside the FlyNet payload")
    args = parser.parse_args(argv)

    hf_token = read_secret("HF_TOKEN")
    github_token = read_secret("GITHUB_ACCESS_TOKEN")
    if not hf_token or not github_token:
        raise RuntimeError("required runtime secrets are unavailable")
    source_dir = args.source_dir.resolve()
    release_dir = args.release_dir.resolve()
    if run(["git", "status", "--porcelain"], cwd=source_dir):
        raise RuntimeError("source checkout is dirty; refusing to publish")
    git_environment = git_env(github_token)
    run(["git", "fetch", "origin", "master"], cwd=source_dir, env=git_environment)
    source_sha = run(["git", "rev-parse", "HEAD"], cwd=source_dir)
    remote_sha = run(["git", "rev-parse", "FETCH_HEAD"], cwd=source_dir)
    if source_sha != remote_sha:
        raise RuntimeError("source checkout is not at the latest pushed master commit")
    home_prefix = private_home_prefix()
    for relative in run(["git", "ls-files"], cwd=source_dir).splitlines():
        candidate = source_dir / relative
        if candidate.is_file():
            contents = candidate.read_text(encoding="utf-8", errors="ignore")
            if (home_prefix and home_prefix in contents) or ABSOLUTE_USER_HOME.search(contents):
                raise RuntimeError("source checkout still contains a private home-path reference")

    card = (source_dir / "docs/huggingface-model-card.md").read_text(encoding="utf-8")
    if not card.startswith("---\n") or "FlyNet" not in card or "ChessFly" in card:
        raise RuntimeError("independent professional FlyNet model card is missing or contaminated")
    config, tensor_shapes, training = validate_release(release_dir)
    tests = test_summary(source_dir)

    from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download  # type: ignore[import-not-found]

    api = HfApi(token=hf_token)
    repo_id = args.repo_id
    if not repo_id:
        identity = api.whoami()
        account = identity.get("name")
        if not account:
            raise RuntimeError("Hub account identity is unavailable")
        repo_id = f"{account}/flychess"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    existing = load_repo_paths(api, repo_id)

    with tempfile.TemporaryDirectory(prefix="flychess-flynet-release-") as temp_dir:
        payload = Path(temp_dir)
        expected_paths, release = stage_payload(
            payload,
            release_dir=release_dir,
            card=card,
            source_sha=source_sha,
            tests=tests,
            config=config,
            tensor_shapes=tensor_shapes,
            training=training,
            dataset_path=args.dataset,
        )
        obsolete = sorted(set(existing) - expected_paths)
        if obsolete and not args.prune:
            raise RuntimeError(
                "target repository contains files outside the FlyNet payload; rerun with --prune after review: "
                + ", ".join(obsolete)
            )
        api.upload_folder(
            folder_path=str(payload),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo="",
            commit_message="publish independent FlyNet release",
        )

    if obsolete:
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=[CommitOperationDelete(path_in_repo=path) for path in obsolete],
            commit_message="remove stale files outside independent FlyNet release",
        )

    final_paths = load_repo_paths(api, repo_id)
    if final_paths != sorted(expected_paths):
        raise RuntimeError(f"unexpected final model file set: {final_paths}")
    info = api.model_info(repo_id, revision="main", files_metadata=True)
    remote_revision = info.sha
    for relative in sorted(expected_paths - {"README.md", ".gitattributes", "model_index.json"}):
        remote_path = hf_hub_download(
            repo_id=repo_id,
            filename=relative,
            revision=remote_revision,
            token=hf_token,
            repo_type="model",
        )
        if relative == "results/release.json":
            # This receipt is intentionally generated after the source commit
            # and remote file set are known, so it has no pre-staged local twin.
            local_digest = None
        elif relative == "flynet-dataset.npz" and args.dataset is not None:
            local_digest = sha256_file(args.dataset)
        elif relative == "flynet-dataset.json":
            local_digest = None
        else:
            local_artifact = release_dir / relative
            local_digest = sha256_file(local_artifact) if local_artifact.is_file() else None
        if local_digest is not None and sha256_file(Path(remote_path)) != local_digest:
            raise RuntimeError(f"Hub checksum mismatch for {relative}")

    print("HF_FLYNET_RELEASE: PASS")
    print("HF_MODEL_REVISION:", remote_revision)
    print("HF_MODEL_FILES:", ",".join(final_paths))
    print("SOURCE_COMMIT:", release["source"]["commit"])
    print("PYTEST:", f"{tests['passed']} passed, {tests['skipped']} skipped")
    print("FLYNET_WEIGHTS_SHA256:", release["artifacts"]["flynet.safetensors"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
