from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from flychess.flynet import ACTION_INDEX, FLYNET_FEATURES, FLYNET_POLICY_ACTIONS
from flychess.flynet_scale import (
    SHARDED_DATASET_SCHEMA,
    SHARD_SCHEMA,
    _iter_batches,
    _record_is_usable,
    load_shard_manifest,
)
from flychess.flynet_training import sha256_file


def _write_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    features = np.zeros((4, FLYNET_FEATURES), dtype=np.float16)
    actions = np.asarray([ACTION_INDEX["e2e4"], ACTION_INDEX["d2d4"], ACTION_INDEX["g1f3"], ACTION_INDEX["c2c4"]], dtype=np.uint16)
    values = np.asarray([0.0, 0.1, -0.2, 0.3], dtype=np.float16)
    data_path = root / "shard-000000.npz"
    np.savez_compressed(data_path, features=features, action_indices=actions, values=values, fens=np.asarray(["a", "b", "c", "d"]))
    record = {
        "schema_version": SHARD_SCHEMA,
        "shard_index": 0,
        "global_start": 0,
        "global_end": 4,
        "samples": 4,
        "features": FLYNET_FEATURES,
        "actions": FLYNET_POLICY_ACTIONS,
        "storage": {"features": "float16", "action_indices": "uint16", "values": "float16"},
        "file": data_path.name,
        "metadata_file": "shard-000000.json",
        "bytes": data_path.stat().st_size,
        "sha256": sha256_file(data_path),
    }
    (root / record["metadata_file"]).write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SHARDED_DATASET_SCHEMA,
        "samples_requested": 4,
        "shard_size": 4,
        "teacher": "Stockfish",
        "teacher_engine": "fixture",
        "teacher_depth": 1,
        "seed": 1,
        "position_sampler": "fixture",
        "min_plies": 0,
        "max_plies": 1,
        "features": FLYNET_FEATURES,
        "actions": FLYNET_POLICY_ACTIONS,
        "shard_count": 1,
        "completed_shards": 1,
        "completed_samples": 4,
        "complete": True,
        "shards": [record],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def test_sharded_fixture_round_trips_and_streams(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    manifest = load_shard_manifest(tmp_path, verify_checksums=True)
    batches = list(
        _iter_batches(
            tmp_path,
            manifest,
            batch_size=2,
            split="train",
            validation_modulus=4,
            shuffle=False,
            seed=1,
            epoch=1,
        )
    )
    assert sum(len(features) for features, _actions, _values in batches) == 3
    assert all(features.dtype == np.float32 for features, _actions, _values in batches)
    validation = list(
        _iter_batches(
            tmp_path,
            manifest,
            batch_size=8,
            split="validation",
            validation_modulus=4,
            shuffle=False,
            seed=1,
            epoch=1,
        )
    )
    assert sum(len(features) for features, _actions, _values in validation) == 1
    assert _record_is_usable(tmp_path, manifest["shards"][0], verify_checksum=True)


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": SHARDED_DATASET_SCHEMA, "complete": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        load_shard_manifest(tmp_path)
