"""Reproducible from-scratch FlyNet data, training, and release helpers.

The teacher is used only to label positions.  No pretrained neural weights,
graph files, or model-specific tensors are read by this module.  A release is
therefore auditable as a Flychess artifact: the dataset, generated graph,
random seeds, optimizer settings, tensor checksum, and held-out metrics travel
together.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from .engine import StockfishEngine
from .flynet import (
    ACTION_INDEX,
    FLYNET_FEATURES,
    FLYNET_POLICY_ACTIONS,
    FLYNET_DEFAULT_EDGES,
    FLYNET_DEFAULT_INPUT_NODES,
    FLYNET_DEFAULT_NODES,
    FLYNET_DEFAULT_READOUT_NODES,
    FLYNET_DEFAULT_STEPS,
    FlyNetDependencyError,
    FlyNetFormatError,
    FlyNetGraph,
    FlyNetModel,
    encode_flynet_board,
    generate_flynet_graph,
    load_graph,
    save_graph,
)


DATASET_SCHEMA = "flynet.dataset/v1"
CONFIG_SCHEMA = "flynet.config/v1"
TRAINING_SCHEMA = "flynet.training/v1"
RELEASE_SCHEMA = "flynet.release/v1"


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise FlyNetDependencyError(
            "FlyNet datasets require the optional 'numpy' dependency; install 'flynet' extras"
        ) from exc
    return np


def _require_torch() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise FlyNetDependencyError(
            "FlyNet training requires the optional 'torch' dependency; install 'flynet' extras"
        ) from exc
    return torch, nn, (DataLoader, TensorDataset)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FlyNetDataset:
    """Validated supervised positions for the independent FlyNet policy."""

    features: Any
    action_indices: Any
    values: Any
    fens: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        np = _require_numpy()
        features = np.asarray(self.features, dtype=np.float32)
        action_indices = np.asarray(self.action_indices, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != FLYNET_FEATURES:
            raise FlyNetFormatError(f"dataset features must have shape [N, {FLYNET_FEATURES}]")
        sample_count = features.shape[0]
        if action_indices.shape != (sample_count,):
            raise FlyNetFormatError("dataset action_indices must align with features")
        if values.shape != (sample_count,):
            raise FlyNetFormatError("dataset values must align with features")
        if np.any(action_indices < 0) or np.any(action_indices >= FLYNET_POLICY_ACTIONS):
            raise FlyNetFormatError("dataset action index is outside FlyNet action space")
        if not np.isfinite(features).all() or not np.isfinite(values).all():
            raise FlyNetFormatError("dataset contains non-finite values")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise FlyNetFormatError("dataset values must be bounded to [-1, 1]")
        fens = tuple(self.fens)
        if len(fens) != sample_count or any(not isinstance(fen, str) or not fen for fen in fens):
            raise FlyNetFormatError("dataset FENs must be non-empty and align with features")
        metadata = dict(self.metadata)
        if metadata and metadata.get("schema_version") not in (None, DATASET_SCHEMA):
            raise FlyNetFormatError("unsupported FlyNet dataset schema")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "action_indices", action_indices)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "fens", fens)
        object.__setattr__(self, "metadata", metadata)

    @property
    def sample_count(self) -> int:
        return int(self.features.shape[0])


def save_dataset(
    dataset: FlyNetDataset,
    path: str | Path,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a portable compressed dataset and a JSON provenance record."""

    np = _require_numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=dataset.features,
        action_indices=dataset.action_indices,
        values=dataset.values,
        fens=np.asarray(dataset.fens),
    )
    metadata = {
        **dataset.metadata,
        "schema_version": DATASET_SCHEMA,
        "samples": dataset.sample_count,
        "features": FLYNET_FEATURES,
        "actions": FLYNET_POLICY_ACTIONS,
    }
    metadata["file"] = path.name
    metadata["sha256"] = sha256_file(path)
    if metadata_path is None:
        metadata_path = path.with_suffix(".json")
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def load_dataset(path: str | Path, metadata_path: str | Path | None = None) -> FlyNetDataset:
    """Load and validate a dataset plus its optional metadata sidecar."""

    np = _require_numpy()
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as values:
            required = {"features", "action_indices", "values", "fens"}
            missing = required.difference(values.files)
            if missing:
                raise FlyNetFormatError(f"dataset is missing arrays: {sorted(missing)}")
            features = values["features"]
            action_indices = values["action_indices"]
            labels = values["values"]
            fens = tuple(str(value) for value in values["fens"].tolist())
    except (OSError, ValueError) as exc:
        if isinstance(exc, FlyNetFormatError):
            raise
        raise FlyNetFormatError(f"could not read FlyNet dataset: {path}") from exc
    if metadata_path is None:
        candidate = path.with_suffix(".json")
        metadata_path = candidate if candidate.is_file() else None
    metadata: dict[str, Any] = {}
    if metadata_path is not None:
        try:
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FlyNetFormatError("could not read FlyNet dataset metadata") from exc
        if not isinstance(metadata, dict):
            raise FlyNetFormatError("FlyNet dataset metadata must be a JSON object")
        if metadata.get("schema_version") not in (None, DATASET_SCHEMA):
            raise FlyNetFormatError("unsupported FlyNet dataset schema")
        expected_sha = metadata.get("sha256")
        if expected_sha is not None and expected_sha != sha256_file(path):
            raise FlyNetFormatError("FlyNet dataset checksum does not match metadata")
    return FlyNetDataset(features, action_indices, labels, fens, metadata)


def _validate_generation_options(count: int, min_plies: int, max_plies: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise FlyNetFormatError("dataset sample count must be an integer >= 2")
    if isinstance(min_plies, bool) or not isinstance(min_plies, int) or min_plies < 0:
        raise FlyNetFormatError("min_plies must be a non-negative integer")
    if isinstance(max_plies, bool) or not isinstance(max_plies, int) or max_plies < min_plies:
        raise FlyNetFormatError("max_plies must be an integer >= min_plies")


def generate_teacher_dataset(
    *,
    engine_path: str = "stockfish",
    depth: int = 3,
    count: int = 10_000,
    seed: int = 20260917,
    min_plies: int = 4,
    max_plies: int = 60,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> FlyNetDataset:
    """Generate legal positions and Stockfish labels with a fixed RNG seed.

    Positions are reached through uniformly sampled legal playouts.  The
    teacher supplies only a move and a side-to-move value; all policy weights
    are initialized later by :func:`train_flynet`.
    """

    _validate_generation_options(count, min_plies, max_plies)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise FlyNetFormatError("Stockfish depth must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FlyNetFormatError("dataset seed must be an integer")
    rng = random.Random(seed)
    features: list[tuple[float, ...]] = []
    action_indices: list[int] = []
    values: list[float] = []
    fens: list[str] = []
    with StockfishEngine(path=engine_path, depth=depth) as engine:
        attempts = 0
        max_attempts = max(count * 8, count + 8)
        while len(features) < count and attempts < max_attempts:
            attempts += 1
            board = chess.Board()
            for _ in range(rng.randint(min_plies, max_plies)):
                if board.is_game_over(claim_draw=True):
                    break
                legal_moves = tuple(board.legal_moves)
                if not legal_moves:
                    break
                board.push(rng.choice(legal_moves))
            if board.is_game_over(claim_draw=True):
                continue
            assessment = engine.assess(board)
            features.append(encode_flynet_board(board))
            action_indices.append(ACTION_INDEX[assessment.move.uci()])
            values.append(assessment.value)
            fens.append(board.fen())
    if len(features) != count:
        raise RuntimeError(f"could generate only {len(features)} of {count} requested positions")
    dataset = FlyNetDataset(
        features=features,
        action_indices=action_indices,
        values=values,
        fens=tuple(fens),
        metadata={
            "seed": seed,
            "teacher": "Stockfish",
            "teacher_engine": engine_path,
            "teacher_depth": depth,
            "position_sampler": "uniform_legal_playout",
            "min_plies": min_plies,
            "max_plies": max_plies,
        },
    )
    if output_path is not None:
        save_dataset(dataset, output_path, metadata_path)
    return dataset


def _read_json_object(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FlyNetFormatError(f"could not read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FlyNetFormatError(f"JSON artifact must be an object: {path}")
    return value


def _set_deterministic_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric_dict(logits: Any, values: Any, actions: Any) -> dict[str, float]:
    _torch, _nn, _data = _require_torch()
    topk = logits.topk(min(5, logits.shape[1]), dim=1).indices
    top1 = (topk[:, 0] == actions).float().mean().item()
    top5 = (topk == actions.unsqueeze(1)).any(dim=1).float().mean().item()
    return {
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
    }


def _evaluate_model(model: FlyNetModel, features: Any, actions: Any, values: Any, *, batch_size: int) -> dict[str, float]:
    torch, _nn, data_classes = _require_torch()
    _loader_type, tensor_dataset_type = data_classes
    loader = _loader_type(tensor_dataset_type(features, actions, values), batch_size=batch_size, shuffle=False)
    policy_logits: list[Any] = []
    predictions: list[Any] = []
    targets: list[Any] = []
    value_targets: list[Any] = []
    model.eval()
    with torch.no_grad():
        for batch_features, batch_actions, batch_values in loader:
            output = model.forward(batch_features)
            policy_logits.append(output.policy_logits.detach().cpu())
            predictions.append(output.value.detach().cpu())
            targets.append(batch_actions.detach().cpu())
            value_targets.append(batch_values.detach().cpu())
    combined_logits = torch.cat(policy_logits, dim=0)
    combined_predictions = torch.cat(predictions, dim=0)
    combined_actions = torch.cat(targets, dim=0)
    combined_values = torch.cat(value_targets, dim=0)
    metrics = _metric_dict(combined_logits, combined_predictions, combined_actions)
    metrics["value_mae"] = (combined_predictions - combined_values).abs().mean().item()
    return metrics


def _artifact_files(output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    return (
        output_dir / "flynet.safetensors",
        output_dir / "flynet-graph.npz",
        output_dir / "flynet-graph.json",
        output_dir / "flynet-config.json",
        output_dir / "results" / "training.json",
    )


def _write_release_manifest(output_dir: Path, files: list[Path], *, metadata: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "artifact": "FlyNet",
        "provenance": "Flychess weights initialized from scratch and trained with Stockfish labels",
        "files": {
            str(path.relative_to(output_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        },
        **metadata,
    }
    manifest_path = output_dir / "results" / "release.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def train_flynet(
    dataset: FlyNetDataset,
    *,
    output_dir: str | Path,
    graph: FlyNetGraph | None = None,
    seed: int = 20260917,
    graph_seed: int | None = None,
    node_count: int = FLYNET_DEFAULT_NODES,
    edge_count: int = FLYNET_DEFAULT_EDGES,
    input_nodes: int = FLYNET_DEFAULT_INPUT_NODES,
    readout_nodes: int = FLYNET_DEFAULT_READOUT_NODES,
    steps: int = FLYNET_DEFAULT_STEPS,
    readout_width: int = 256,
    epochs: int = 12,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    value_weight: float = 0.25,
    validation_fraction: float = 0.15,
    device: str | None = None,
) -> dict[str, Any]:
    """Train FlyNet from random initialization and write a release bundle."""

    torch, _nn, data_classes = _require_torch()
    np = _require_numpy()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise FlyNetFormatError("training seed must be an integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise FlyNetFormatError("epochs must be an integer >= 1")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise FlyNetFormatError("batch_size must be an integer >= 1")
    if not math_is_finite_positive(learning_rate):
        raise FlyNetFormatError("learning_rate must be finite and positive")
    if not math_is_finite_nonnegative(value_weight):
        raise FlyNetFormatError("value_weight must be finite and non-negative")
    if not 0.0 < validation_fraction < 0.5:
        raise FlyNetFormatError("validation_fraction must be between 0 and 0.5")
    if dataset.sample_count < 2:
        raise FlyNetFormatError("at least two samples are required for training")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = graph or generate_flynet_graph(
        node_count=node_count,
        edge_count=edge_count,
        input_nodes=input_nodes,
        readout_nodes=readout_nodes,
        seed=seed if graph_seed is None else graph_seed,
    )
    graph_seed = graph.seed

    _set_deterministic_seed(torch, seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FlyNetModel(graph, steps=steps, readout_width=readout_width, device=selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    cross_entropy = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    features = torch.from_numpy(dataset.features)
    actions = torch.from_numpy(dataset.action_indices)
    values = torch.from_numpy(dataset.values)
    permutation = np.random.default_rng(seed).permutation(dataset.sample_count)
    validation_count = max(1, int(round(dataset.sample_count * validation_fraction)))
    train_indices = permutation[validation_count:]
    validation_indices = permutation[:validation_count]
    if not len(train_indices):
        raise FlyNetFormatError("validation split leaves no training samples")
    train_index_tensor = torch.from_numpy(train_indices)
    validation_index_tensor = torch.from_numpy(validation_indices)
    train_tensors = tuple(tensor[train_index_tensor] for tensor in (features, actions, values))
    validation_tensors = tuple(tensor[validation_index_tensor] for tensor in (features, actions, values))
    loader_type, tensor_dataset_type = data_classes
    loader = loader_type(
        tensor_dataset_type(*train_tensors),
        batch_size=min(batch_size, len(train_indices)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_features, batch_actions, batch_values in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model.forward(batch_features)
            loss = cross_entropy(output.policy_logits, batch_actions) + value_weight * mse(output.value, batch_values)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
            optimizer.step()
            count = batch_features.shape[0]
            running_loss += float(loss.detach().cpu().item()) * count
            seen += count
        validation_metrics = _evaluate_model(
            model,
            validation_tensors[0],
            validation_tensors[1],
            validation_tensors[2],
            batch_size=min(batch_size, len(validation_indices)),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, seen),
                "validation": validation_metrics,
            }
        )

    weights_path, graph_path, graph_metadata_path, config_path, training_path = _artifact_files(output_dir)
    save_graph(graph, graph_path, graph_metadata_path)
    model.save_safetensors(weights_path)
    training_result = {
        "schema_version": TRAINING_SCHEMA,
        "seed": seed,
        "graph_seed": graph_seed,
        "dataset": dataset.metadata,
        "samples": dataset.sample_count,
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "device": str(model.device),
        "optimizer": "AdamW",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "value_weight": value_weight,
        "history": history,
        "final_validation": history[-1]["validation"],
    }
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(json.dumps(training_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "schema_version": CONFIG_SCHEMA,
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
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _write_release_manifest(
        output_dir,
        [weights_path, graph_path, graph_metadata_path, config_path, training_path],
        metadata={
            "config": config,
            "training": {
                "seed": seed,
                "dataset_samples": dataset.sample_count,
                "final_validation": history[-1]["validation"],
            },
        },
    )
    return {"output_dir": str(output_dir), "config": config, "training": training_result, "release": manifest}


def math_is_finite_positive(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0.0
    )


def math_is_finite_nonnegative(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0.0
    )


def load_flynet_model(
    weights_path: str | Path,
    graph_path: str | Path,
    *,
    graph_metadata_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str = "cpu",
) -> FlyNetModel:
    """Load an independent FlyNet release after validating its contracts."""

    graph = load_graph(graph_path, graph_metadata_path)
    config: dict[str, Any] = {}
    if config_path is not None:
        config = _read_json_object(config_path)
        if config.get("schema_version") not in (None, CONFIG_SCHEMA):
            raise FlyNetFormatError("unsupported FlyNet config schema")
        if config.get("features", FLYNET_FEATURES) != FLYNET_FEATURES:
            raise FlyNetFormatError("FlyNet config feature count does not match runtime")
        if config.get("actions", FLYNET_POLICY_ACTIONS) != FLYNET_POLICY_ACTIONS:
            raise FlyNetFormatError("FlyNet config action count does not match runtime")
        configured_graph = config.get("graph")
        if configured_graph is not None and configured_graph != graph.summary():
            raise FlyNetFormatError("FlyNet config graph summary does not match graph artifact")
    steps = int(config.get("steps", FLYNET_DEFAULT_STEPS))
    readout_width = int(config.get("readout_width", 256))
    model = FlyNetModel(graph, steps=steps, readout_width=readout_width, device=device)
    model.load_safetensors(weights_path)
    return model


__all__ = [
    "CONFIG_SCHEMA",
    "DATASET_SCHEMA",
    "FlyNetDataset",
    "RELEASE_SCHEMA",
    "TRAINING_SCHEMA",
    "generate_teacher_dataset",
    "load_dataset",
    "load_flynet_model",
    "save_dataset",
    "sha256_file",
    "train_flynet",
]
