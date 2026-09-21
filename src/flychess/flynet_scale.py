"""Resumable sharded FlyNet datasets and streamed large-run training.

The small ``flynet_training`` helpers intentionally keep a complete dataset in
memory.  That is useful for tests and smoke runs, but it does not scale to
millions of Stockfish-labelled positions.  This module stores independently
verified compressed shards and trains one shard at a time.

Shards are deterministic by global sample index.  A failed Colab runtime can
therefore resume without regenerating completed shards, while the manifest
records the exact teacher and sampler configuration used for the run.
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator

import chess

from .engine import StockfishEngine
from .flynet import (
    ACTION_INDEX,
    FLYNET_FEATURES,
    FLYNET_POLICY_ACTIONS,
    FlyNetFormatError,
    FlyNetGraph,
    FlyNetModel,
    generate_flynet_graph,
    save_graph,
)
from .flynet_training import sha256_file


SHARDED_DATASET_SCHEMA = "flynet.sharded_dataset/v1"
SHARD_SCHEMA = "flynet.shard/v1"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("large FlyNet runs require numpy") from exc
    return np


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("large FlyNet runs require torch") from exc
    return torch, torch.nn


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_positive_int(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FlyNetFormatError(f"{name} must be an integer >= {minimum}")


def _validate_generation_options(samples: int, shard_size: int, min_plies: int, max_plies: int) -> None:
    _validate_positive_int("samples", samples, minimum=2)
    _validate_positive_int("shard_size", shard_size)
    if isinstance(min_plies, bool) or not isinstance(min_plies, int) or min_plies < 0:
        raise FlyNetFormatError("min_plies must be a non-negative integer")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies < min_plies:
        raise FlyNetFormatError("max_plies must be an integer >= min_plies")


def _sample_board(*, seed: int, global_index: int, min_plies: int, max_plies: int) -> chess.Board:
    """Build one deterministic legal-playout position for a global index."""

    # Independent per-index RNG streams make shard boundaries and resume state
    # irrelevant to the generated position sequence.
    for attempt in range(32):
        rng = random.Random(seed + global_index * 1_000_003 + attempt * 9_176)
        board = chess.Board()
        for _ in range(rng.randint(min_plies, max_plies)):
            if board.is_game_over(claim_draw=True):
                break
            legal_moves = tuple(board.legal_moves)
            if not legal_moves:
                break
            board.push(rng.choice(legal_moves))
        if not board.is_game_over(claim_draw=True):
            return board
    raise RuntimeError(f"could not sample a non-terminal position for index {global_index}")


def _shard_names(index: int) -> tuple[str, str]:
    stem = f"shard-{index:06d}"
    return f"{stem}.npz", f"{stem}.json"


def _record_is_usable(root: Path, record: dict[str, Any], *, verify_checksum: bool = False) -> bool:
    try:
        data_path = root / str(record["file"])
        metadata_path = root / str(record["metadata_file"])
        if not data_path.is_file() or not metadata_path.is_file():
            return False
        if data_path.stat().st_size != int(record["bytes"]):
            return False
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata != record:
            return False
        if verify_checksum and sha256_file(data_path) != str(record["sha256"]):
            return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_shard(
    *,
    root: Path,
    shard_index: int,
    global_start: int,
    samples: int,
    engine_path: str,
    depth: int,
    seed: int,
    min_plies: int,
    max_plies: int,
) -> dict[str, Any]:
    """Generate one shard in a worker and publish it with an atomic boundary."""

    np = _require_numpy()
    _validate_positive_int("depth", depth)
    data_name, metadata_name = _shard_names(shard_index)
    data_path = root / data_name
    metadata_path = root / metadata_name
    root.mkdir(parents=True, exist_ok=True)

    features = np.empty((samples, FLYNET_FEATURES), dtype=np.float16)
    action_indices = np.empty(samples, dtype=np.uint16)
    values = np.empty(samples, dtype=np.float16)
    fens: list[str] = []
    with StockfishEngine(path=engine_path, depth=depth) as engine:
        for offset in range(samples):
            global_index = global_start + offset
            board = _sample_board(
                seed=seed,
                global_index=global_index,
                min_plies=min_plies,
                max_plies=max_plies,
            )
            assessment = engine.assess(board)
            features[offset] = np.asarray(
                # Storing features as float16 halves I/O while preserving the
                # encoder's binary flags and bounded scalar channels.
                tuple(float(value) for value in _encode_board(board)),
                dtype=np.float16,
            )
            action_indices[offset] = ACTION_INDEX[assessment.move.uci()]
            values[offset] = assessment.value
            fens.append(board.fen())

    temporary_data = root / f".{data_name}.{os.getpid()}.part"
    try:
        np.savez_compressed(
            temporary_data,
            features=features,
            action_indices=action_indices,
            values=values,
            fens=np.asarray(fens),
        )
        # numpy appends .npz when the path does not have that suffix.
        actual_temporary_data = temporary_data.with_name(temporary_data.name + ".npz")
        digest = sha256_file(actual_temporary_data)
        byte_count = actual_temporary_data.stat().st_size
        os.replace(actual_temporary_data, data_path)
    finally:
        temporary_data.unlink(missing_ok=True)
        temporary_data.with_name(temporary_data.name + ".npz").unlink(missing_ok=True)

    record: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "shard_index": shard_index,
        "global_start": global_start,
        "global_end": global_start + samples,
        "samples": samples,
        "features": FLYNET_FEATURES,
        "actions": FLYNET_POLICY_ACTIONS,
        "storage": {
            "features": "float16",
            "action_indices": "uint16",
            "values": "float16",
        },
        "file": data_name,
        "metadata_file": metadata_name,
        "bytes": byte_count,
        "sha256": digest,
    }
    _write_json_atomic(metadata_path, record)
    return record


def _encode_board(board: chess.Board) -> tuple[float, ...]:
    # Imported lazily here to keep worker startup small and to ensure the
    # exact public encoder remains the single source of feature semantics.
    from .flynet import encode_flynet_board

    return encode_flynet_board(board)


def _manifest_config(
    *,
    samples: int,
    shard_size: int,
    engine_path: str,
    depth: int,
    seed: int,
    min_plies: int,
    max_plies: int,
) -> dict[str, Any]:
    return {
        "schema_version": SHARDED_DATASET_SCHEMA,
        "samples_requested": samples,
        "shard_size": shard_size,
        "teacher": "Stockfish",
        "teacher_engine": engine_path,
        "teacher_depth": depth,
        "seed": seed,
        "position_sampler": "uniform_legal_playout_per_global_index",
        "min_plies": min_plies,
        "max_plies": max_plies,
        "features": FLYNET_FEATURES,
        "actions": FLYNET_POLICY_ACTIONS,
    }


def _new_manifest(config: dict[str, Any], shard_count: int) -> dict[str, Any]:
    return {
        **config,
        "shard_count": shard_count,
        "completed_shards": 0,
        "completed_samples": 0,
        "complete": False,
        "shards": [],
    }


def generate_teacher_shards(
    *,
    root: str | Path,
    engine_path: str = "stockfish",
    depth: int = 3,
    samples: int = 50_000_000,
    shard_size: int = 100_000,
    seed: int = 20260917,
    min_plies: int = 4,
    max_plies: int = 60,
    workers: int = 1,
    resume: bool = True,
    progress: Callable[[dict[str, Any], int, int], None] | None = None,
) -> dict[str, Any]:
    """Generate a resumable, checksum-recorded Stockfish dataset directory."""

    _validate_generation_options(samples, shard_size, min_plies, max_plies)
    _validate_positive_int("depth", depth)
    _validate_positive_int("workers", workers)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FlyNetFormatError("dataset seed must be an integer")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    shard_count = math.ceil(samples / shard_size)
    config = _manifest_config(
        samples=samples,
        shard_size=shard_size,
        engine_path=engine_path,
        depth=depth,
        seed=seed,
        min_plies=min_plies,
        max_plies=max_plies,
    )
    manifest_path = root / "manifest.json"
    manifest = _new_manifest(config, shard_count)
    if resume and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FlyNetFormatError(f"could not read sharded dataset manifest: {manifest_path}") from exc
        if not isinstance(existing, dict) or any(existing.get(key) != value for key, value in config.items()):
            raise FlyNetFormatError("existing sharded dataset manifest does not match requested configuration")
        manifest = existing

    records_by_index: dict[int, dict[str, Any]] = {}
    for raw_record in manifest.get("shards", []):
        if isinstance(raw_record, dict) and _record_is_usable(root, raw_record):
            records_by_index[int(raw_record["shard_index"])] = raw_record

    def refresh_manifest() -> None:
        ordered = [records_by_index[index] for index in sorted(records_by_index)]
        manifest.update(
            {
                "completed_shards": len(ordered),
                "completed_samples": sum(int(record["samples"]) for record in ordered),
                "complete": len(ordered) == shard_count,
                "shards": ordered,
            }
        )
        _write_json_atomic(manifest_path, manifest)

    refresh_manifest()
    missing = [
        {
            "root": str(root),
            "shard_index": index,
            "global_start": index * shard_size,
            "samples": min(shard_size, samples - index * shard_size),
            "engine_path": engine_path,
            "depth": depth,
            "seed": seed,
            "min_plies": min_plies,
            "max_plies": max_plies,
        }
        for index in range(shard_count)
        if index not in records_by_index
    ]

    if workers == 1:
        for spec in missing:
            record = _write_shard(**{**spec, "root": Path(spec["root"])})
            records_by_index[int(record["shard_index"])] = record
            refresh_manifest()
            if progress is not None:
                progress(record, len(records_by_index), shard_count)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_write_shard, **{**spec, "root": Path(spec["root"])}): spec for spec in missing}
            for future in as_completed(futures):
                record = future.result()
                records_by_index[int(record["shard_index"])] = record
                refresh_manifest()
                if progress is not None:
                    progress(record, len(records_by_index), shard_count)
    return manifest


def load_shard_manifest(root: str | Path, *, verify_checksums: bool = False) -> dict[str, Any]:
    """Load and validate a complete sharded dataset manifest."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlyNetFormatError(f"could not read sharded dataset manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SHARDED_DATASET_SCHEMA:
        raise FlyNetFormatError("unsupported sharded FlyNet dataset schema")
    if not manifest.get("complete"):
        raise FlyNetFormatError("sharded FlyNet dataset is incomplete")
    shards = manifest.get("shards")
    expected_count = int(manifest.get("shard_count", 0))
    if not isinstance(shards, list) or len(shards) != expected_count:
        raise FlyNetFormatError("sharded FlyNet manifest has an incomplete shard list")
    for index, record in enumerate(shards):
        if not isinstance(record, dict) or int(record.get("shard_index", -1)) != index:
            raise FlyNetFormatError("sharded FlyNet manifest shard order is invalid")
        if not _record_is_usable(root, record, verify_checksum=verify_checksums):
            raise FlyNetFormatError(f"sharded FlyNet artifact failed verification: shard {index}")
    return manifest


def _validation_modulus(validation_fraction: float) -> int:
    if not isinstance(validation_fraction, (int, float)) or isinstance(validation_fraction, bool):
        raise FlyNetFormatError("validation_fraction must be numeric")
    if not 0.0 < float(validation_fraction) < 0.5:
        raise FlyNetFormatError("validation_fraction must be between 0 and 0.5")
    return max(2, int(round(1.0 / float(validation_fraction))))


def _iter_batches(
    root: Path,
    manifest: dict[str, Any],
    *,
    batch_size: int,
    split: str,
    validation_modulus: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> Iterator[tuple[Any, Any, Any]]:
    np = _require_numpy()
    _validate_positive_int("batch_size", batch_size)
    if split not in {"train", "validation"}:
        raise FlyNetFormatError("split must be 'train' or 'validation'")
    for record in manifest["shards"]:
        data_path = root / str(record["file"])
        with np.load(data_path, allow_pickle=False) as arrays:
            features = arrays["features"]
            actions = arrays["action_indices"]
            values = arrays["values"]
            if features.ndim != 2 or features.shape[1] != FLYNET_FEATURES:
                raise FlyNetFormatError(f"invalid feature matrix in {data_path}")
            count = int(features.shape[0])
            if count != int(record["samples"]):
                raise FlyNetFormatError(f"shard sample count mismatch in {data_path}")
            global_indices = np.arange(int(record["global_start"]), int(record["global_end"]), dtype=np.int64)
            validation_mask = (global_indices % validation_modulus) == 0
            selected = np.flatnonzero(validation_mask if split == "validation" else ~validation_mask)
            if shuffle:
                np.random.default_rng(seed + epoch * 1_000_003 + int(record["shard_index"])).shuffle(selected)
            for start in range(0, len(selected), batch_size):
                indices = selected[start : start + batch_size]
                yield (
                    np.asarray(features[indices], dtype=np.float32),
                    np.asarray(actions[indices], dtype=np.int64),
                    np.asarray(values[indices], dtype=np.float32),
                )


def _evaluate_stream(
    model: FlyNetModel,
    batches: Iterator[tuple[Any, Any, Any]],
    *,
    limit: int | None = None,
) -> dict[str, float]:
    torch, _nn = _require_torch()
    top1 = 0
    top5 = 0
    seen = 0
    value_error = 0.0
    model.eval()
    with torch.no_grad():
        for features, actions, values in batches:
            if limit is not None and seen >= limit:
                break
            if limit is not None and seen + len(features) > limit:
                keep = limit - seen
                features, actions, values = features[:keep], actions[:keep], values[:keep]
            batch_features = torch.from_numpy(features).to(model.device)
            batch_actions = torch.from_numpy(actions).to(model.device)
            batch_values = torch.from_numpy(values).to(model.device)
            output = model.forward(batch_features)
            ranked = output.policy_logits.topk(min(5, output.policy_logits.shape[1]), dim=1).indices
            top1 += int((ranked[:, 0] == batch_actions).sum().item())
            top5 += int((ranked == batch_actions.unsqueeze(1)).any(dim=1).sum().item())
            value_error += float((output.value - batch_values).abs().sum().item())
            seen += len(features)
    return {
        "samples": float(seen),
        "top1_accuracy": top1 / seen if seen else 0.0,
        "top5_accuracy": top5 / seen if seen else 0.0,
        "value_mae": value_error / seen if seen else 0.0,
    }


def _write_progress(
    output_dir: Path,
    *,
    schema_version: str,
    status: str,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    path = output_dir / "results" / "training.json"
    _write_json_atomic(
        path,
        {
            "schema_version": schema_version,
            "status": status,
            "config": config,
            "history": history,
        },
    )


def train_flynet_sharded(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    seed: int = 20260917,
    graph_seed: int | None = None,
    node_count: int = 2_048,
    edge_count: int = 64_000,
    input_nodes: int = 256,
    readout_nodes: int = 320,
    steps: int = 6,
    readout_width: int = 256,
    epochs: int = 1,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    value_weight: float = 0.25,
    validation_fraction: float = 0.01,
    validation_limit: int = 100_000,
    device: str | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Train from one verified shard at a time and write a release bundle."""

    torch, _nn = _require_torch()
    _validate_positive_int("epochs", epochs)
    _validate_positive_int("batch_size", batch_size)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise FlyNetFormatError("training seed must be an integer")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise FlyNetFormatError("learning_rate must be finite and positive")
    if not math.isfinite(float(value_weight)) or value_weight < 0:
        raise FlyNetFormatError("value_weight must be finite and non-negative")
    validation_modulus = _validation_modulus(validation_fraction)
    dataset_root = Path(dataset_root)
    manifest = load_shard_manifest(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = generate_flynet_graph(
        node_count=node_count,
        edge_count=edge_count,
        input_nodes=input_nodes,
        readout_nodes=readout_nodes,
        seed=seed if graph_seed is None else graph_seed,
    )
    graph_seed = graph.seed
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FlyNetModel(graph, steps=steps, readout_width=readout_width, device=selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    checkpoint_path = output_dir / "checkpoints" / "latest.pt"
    history: list[dict[str, Any]] = []
    start_epoch = 1
    if resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=str(model.device))
        if checkpoint.get("graph") != graph.summary():
            raise FlyNetFormatError("checkpoint graph does not match requested training graph")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    training_config = {
        "dataset_manifest": f"{dataset_root.name}/manifest.json",
        "dataset_samples": int(manifest["samples_requested"]),
        "validation_fraction_requested": float(validation_fraction),
        "validation_modulus": validation_modulus,
        "seed": seed,
        "graph_seed": graph_seed,
        "nodes": node_count,
        "edges": edge_count,
        "input_nodes": input_nodes,
        "readout_nodes": readout_nodes,
        "steps": steps,
        "readout_width": readout_width,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_weight": value_weight,
        "device": str(model.device),
        "initialization": "random_seeded",
        "pretrained_weights_used": False,
    }
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, actions, values in _iter_batches(
            dataset_root,
            manifest,
            batch_size=batch_size,
            split="train",
            validation_modulus=validation_modulus,
            shuffle=True,
            seed=seed,
            epoch=epoch,
        ):
            batch_features = torch.from_numpy(features).to(model.device)
            batch_actions = torch.from_numpy(actions).to(model.device)
            batch_values = torch.from_numpy(values).to(model.device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward(batch_features)
            loss = cross_entropy(output.policy_logits, batch_actions) + value_weight * mse(output.value, batch_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
            optimizer.step()
            count = len(features)
            running_loss += float(loss.detach().cpu().item()) * count
            seen += count
        validation = _evaluate_stream(
            model,
            _iter_batches(
                dataset_root,
                manifest,
                batch_size=batch_size,
                split="validation",
                validation_modulus=validation_modulus,
                shuffle=False,
                seed=seed,
                epoch=epoch,
            ),
            limit=validation_limit if validation_limit > 0 else None,
        )
        record = {
            "epoch": epoch,
            "train_samples": seen,
            "train_loss": running_loss / max(1, seen),
            "validation": validation,
        }
        history.append(record)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "graph": graph.summary(), "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history}, checkpoint_path)
        _write_progress(
            output_dir,
            schema_version="flynet.large_training/v1",
            status="in_progress" if epoch < epochs else "complete",
            history=history,
            config=training_config,
        )

    from .flynet_training import sha256_file as _sha256_file

    weights_path = output_dir / "flynet.safetensors"
    graph_path = output_dir / "flynet-graph.npz"
    graph_metadata_path = output_dir / "flynet-graph.json"
    config_path = output_dir / "flynet-config.json"
    training_path = output_dir / "results" / "training.json"
    save_graph(graph, graph_path, graph_metadata_path)
    model.save_safetensors(weights_path)
    config = {
        "schema_version": "flynet.config/v1",
        "model_type": "flynet.signed_recurrent_graph",
        "features": FLYNET_FEATURES,
        "actions": FLYNET_POLICY_ACTIONS,
        "value_representation": "scalar_side_to_move_tanh",
        "steps": steps,
        "readout_width": readout_width,
        "graph": graph.summary(),
        "weights": weights_path.name,
        "graph_file": graph_path.name,
        "graph_metadata": graph_metadata_path.name,
        "training_file": str(training_path.relative_to(output_dir)),
        "initialization": "random_seeded",
        "pretrained_weights_used": False,
        "teacher_labels": "Stockfish move and bounded side-to-move value",
        "dataset_manifest": f"{dataset_root.name}/manifest.json",
    }
    _write_json_atomic(config_path, config)
    release = {
        "schema_version": "flynet.release/v1",
        "artifact": "FlyNet",
        "provenance": "Flychess weights initialized from scratch and trained with Stockfish labels",
        "dataset": {
            "manifest": f"{dataset_root.name}/manifest.json",
            "manifest_sha256": _sha256_file(dataset_root / "manifest.json"),
            "samples": int(manifest["samples_requested"]),
        },
        "config": config,
        "training": {
            "seed": seed,
            "dataset_samples": int(manifest["samples_requested"]),
            "final_validation": history[-1]["validation"] if history else {},
        },
        "files": {
            str(path.relative_to(output_dir)): {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in (weights_path, graph_path, graph_metadata_path, config_path, training_path)
        },
    }
    _write_json_atomic(output_dir / "results" / "release.json", release)
    return {"output_dir": str(output_dir), "config": config, "training": history, "release": release}


__all__ = [
    "SHARDED_DATASET_SCHEMA",
    "SHARD_SCHEMA",
    "generate_teacher_shards",
    "load_shard_manifest",
    "train_flynet_sharded",
]
