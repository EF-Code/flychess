"""Flychess's from-scratch graph policy and artifact contract.

This module is deliberately independent of the public ChessFly artifact.  A
``FlyNetGraph`` is generated locally from a seed, its signed recurrent edge
gains are trainable parameters, and the policy/value heads are initialized and
trained by the Flychess pipeline.  The graph is brain-inspired in its region
layout but is not a biological connectome reconstruction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from .brain import DecisionReadout, MoveCandidate


FLYNET_FEATURES = 851
FLYNET_VALUE_BINS = 3
FLYNET_DEFAULT_STEPS = 6
FLYNET_DEFAULT_NODES = 2_048
FLYNET_DEFAULT_EDGES = 64_000
FLYNET_DEFAULT_INPUT_NODES = 256
FLYNET_DEFAULT_READOUT_NODES = 320
FLYNET_REGIONS = ("optic", "association", "memory", "action", "value")
PROMOTION_TYPES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


class FlyNetDependencyError(RuntimeError):
    """Raised when optional FlyNet training/runtime dependencies are missing."""


class FlyNetFormatError(ValueError):
    """Raised when an independent FlyNet graph or artifact is malformed."""


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise FlyNetDependencyError(
            "FlyNet requires the optional 'torch' dependency; install 'flynet' extras"
        ) from exc
    return torch, nn


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise FlyNetDependencyError(
            "FlyNet graph artifacts require the optional 'numpy' dependency"
        ) from exc
    return np


def _promotion_uci(from_square: int, to_square: int, promotion: chess.PieceType) -> str:
    return chess.Move(from_square, to_square, promotion=promotion).uci()


def build_action_space() -> tuple[str, ...]:
    """Build a deterministic, color-preserving vocabulary of legal UCI shapes."""

    actions: set[str] = set()
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            if from_square != to_square:
                actions.add(chess.Move(from_square, to_square).uci())
        from_rank = chess.square_rank(from_square)
        if from_rank in (1, 6):
            target_rank = 0 if from_rank == 1 else 7
            for target_file in range(8):
                to_square = chess.square(target_file, target_rank)
                for promotion in PROMOTION_TYPES:
                    actions.add(_promotion_uci(from_square, to_square, promotion))
    return tuple(sorted(actions))


ACTION_SPACE = build_action_space()
ACTION_INDEX = {uci: index for index, uci in enumerate(ACTION_SPACE)}
FLYNET_POLICY_ACTIONS = len(ACTION_SPACE)


def encode_flynet_board(board: chess.Board) -> tuple[float, ...]:
    """Encode a board into FlyNet's independent 851-feature sensory contract."""

    features = [0.0] * FLYNET_FEATURES
    offset = 0
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                features[offset] = float(
                    piece is not None and piece.color == color and piece.piece_type == piece_type
                )
                offset += 1

    features[offset] = 1.0 if board.turn == chess.WHITE else -1.0
    offset += 1
    for color in (chess.WHITE, chess.BLACK):
        features[offset] = float(board.has_kingside_castling_rights(color))
        features[offset + 1] = float(board.has_queenside_castling_rights(color))
        offset += 2

    if board.ep_square is not None:
        features[offset + board.ep_square] = 1.0
    offset += 64
    features[offset] = min(board.halfmove_clock, 100) / 100.0
    features[offset + 1] = min(board.fullmove_number, 200) / 200.0
    offset += 2

    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            features[offset] = min(len(board.pieces(piece_type, color)), 8) / 8.0
            offset += 1
    if offset != FLYNET_FEATURES:
        raise FlyNetFormatError(f"FlyNet feature encoder produced {offset} values")
    return tuple(features)


def action_index(move: chess.Move) -> int:
    """Return the fixed action index for a legal chess move."""

    try:
        return ACTION_INDEX[move.uci()]
    except KeyError as exc:
        raise FlyNetFormatError(f"move is outside FlyNet action space: {move.uci()}") from exc


@dataclass(frozen=True, slots=True)
class FlyNetGraph:
    """Validated sparse signed graph used by the independent FlyNet core."""

    node_count: int
    edge_source: tuple[int, ...]
    edge_target: tuple[int, ...]
    edge_sign: tuple[float, ...]
    input_nodes: tuple[int, ...]
    readout_nodes: tuple[int, ...]
    regions: tuple[str, ...]
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int) or self.node_count < 8:
            raise FlyNetFormatError("node_count must be an integer >= 8")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise FlyNetFormatError("FlyNet graph seed must be an integer")
        edge_source = tuple(self.edge_source)
        edge_target = tuple(self.edge_target)
        edge_sign = tuple(float(value) for value in self.edge_sign)
        if not edge_source or len(edge_source) != len(edge_target) or len(edge_source) != len(edge_sign):
            raise FlyNetFormatError("FlyNet graph edges must be non-empty and aligned")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < self.node_count
            for value in (*edge_source, *edge_target)
        ):
            raise FlyNetFormatError("FlyNet edge indices are out of bounds")
        if any(not math.isfinite(value) or value not in (-1.0, 1.0) for value in edge_sign):
            raise FlyNetFormatError("FlyNet edge signs must be finite +/-1 values")
        input_nodes = tuple(self.input_nodes)
        readout_nodes = tuple(self.readout_nodes)
        for label, values in (("input", input_nodes), ("readout", readout_nodes)):
            if not values or len(set(values)) != len(values):
                raise FlyNetFormatError(f"FlyNet {label} nodes must be unique and non-empty")
            if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.node_count for value in values):
                raise FlyNetFormatError(f"FlyNet {label} node index is out of bounds")
        regions = tuple(self.regions)
        if len(regions) != self.node_count or any(not isinstance(value, str) or not value for value in regions):
            raise FlyNetFormatError("FlyNet regions must name every node")
        if any(value not in FLYNET_REGIONS for value in regions):
            raise FlyNetFormatError("FlyNet graph contains an unknown region")
        object.__setattr__(self, "edge_source", edge_source)
        object.__setattr__(self, "edge_target", edge_target)
        object.__setattr__(self, "edge_sign", edge_sign)
        object.__setattr__(self, "input_nodes", input_nodes)
        object.__setattr__(self, "readout_nodes", readout_nodes)
        object.__setattr__(self, "regions", regions)

    @property
    def edge_count(self) -> int:
        return len(self.edge_source)

    def summary(self) -> dict[str, Any]:
        return {
            "nodes": self.node_count,
            "edges": self.edge_count,
            "inputs": len(self.input_nodes),
            "readout": len(self.readout_nodes),
            "regions": {region: self.regions.count(region) for region in FLYNET_REGIONS},
            "seed": self.seed,
        }


def generate_flynet_graph(
    *,
    node_count: int = FLYNET_DEFAULT_NODES,
    edge_count: int = FLYNET_DEFAULT_EDGES,
    input_nodes: int = FLYNET_DEFAULT_INPUT_NODES,
    readout_nodes: int = FLYNET_DEFAULT_READOUT_NODES,
    seed: int = 20260917,
) -> FlyNetGraph:
    """Generate a deterministic, region-structured signed graph from scratch."""

    np = _require_numpy()
    integer_arguments = {
        "node_count": node_count,
        "edge_count": edge_count,
        "input_nodes": input_nodes,
        "readout_nodes": readout_nodes,
        "seed": seed,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_arguments.values()):
        raise FlyNetFormatError("FlyNet graph dimensions and seed must be integers")
    if node_count < 8 or edge_count < 1 or input_nodes < 1 or readout_nodes < 1:
        raise FlyNetFormatError("FlyNet graph dimensions are too small")
    if input_nodes + readout_nodes >= node_count:
        raise FlyNetFormatError("FlyNet graph needs at least one non-port node")
    if edge_count > node_count * (node_count - 1):
        raise FlyNetFormatError("FlyNet edge count exceeds the directed graph capacity")

    readout_action_count = readout_nodes // 2
    readout_value_count = readout_nodes - readout_action_count
    internal_nodes = node_count - input_nodes - readout_nodes
    association_count = max(1, (internal_nodes * 3) // 5)
    memory_count = internal_nodes - association_count
    if memory_count < 0:
        association_count = internal_nodes
        memory_count = 0

    regions: list[str] = (
        ["optic"] * input_nodes
        + ["association"] * association_count
        + ["memory"] * memory_count
        + ["action"] * readout_action_count
        + ["value"] * readout_value_count
    )
    if len(regions) != node_count:
        raise FlyNetFormatError("FlyNet region layout does not cover every node")

    rng = np.random.default_rng(seed)

    def source_pool_for(target: int) -> Any:
        target_region = regions[target]
        if target_region == "optic":
            # Sensory nodes can receive recurrent feedback from the entire
            # graph; this keeps the port a node subset rather than a second
            # hidden input projection.
            return np.arange(node_count)
        if target_region in ("action", "value"):
            # Output ports read from the internal circuit, never from another
            # output port. The decoder remains an explicit, inspectable head.
            return np.arange(input_nodes, node_count - readout_nodes)
        return np.arange(node_count)

    max_edges = 0
    for target in range(node_count):
        pool_size = len(source_pool_for(target))
        max_edges += pool_size - (1 if target in source_pool_for(target) else 0)
    if edge_count > max_edges:
        raise FlyNetFormatError("FlyNet edge count exceeds the region-constrained graph capacity")

    edges: set[tuple[int, int]] = set()
    while len(edges) < edge_count:
        target = int(rng.integers(0, node_count))
        source_pool = source_pool_for(target)
        source = int(source_pool[int(rng.integers(0, len(source_pool)))])
        if source == target:
            continue
        edges.add((source, target))

    ordered_edges = sorted(edges, key=lambda edge: (edge[1], edge[0]))
    signs = tuple(1.0 if value else -1.0 for value in rng.integers(0, 2, size=len(ordered_edges)))
    readout_start = node_count - readout_nodes
    return FlyNetGraph(
        node_count=node_count,
        edge_source=tuple(source for source, _target in ordered_edges),
        edge_target=tuple(target for _source, target in ordered_edges),
        edge_sign=signs,
        input_nodes=tuple(range(input_nodes)),
        readout_nodes=tuple(range(readout_start, node_count)),
        regions=tuple(regions),
        seed=seed,
    )


def save_graph(graph: FlyNetGraph, path: str | Path, metadata_path: str | Path | None = None) -> None:
    """Write graph arrays and a portable metadata record."""

    np = _require_numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        node_count=np.asarray(graph.node_count, dtype=np.int64),
        seed=np.asarray(graph.seed, dtype=np.int64),
        edge_source=np.asarray(graph.edge_source, dtype=np.int64),
        edge_target=np.asarray(graph.edge_target, dtype=np.int64),
        edge_sign=np.asarray(graph.edge_sign, dtype=np.float32),
        input_nodes=np.asarray(graph.input_nodes, dtype=np.int64),
        readout_nodes=np.asarray(graph.readout_nodes, dtype=np.int64),
        regions=np.asarray(graph.regions),
    )
    if metadata_path is not None:
        metadata = {"schema_version": "flynet.graph/v1", **graph.summary()}
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_graph(path: str | Path, metadata_path: str | Path | None = None) -> FlyNetGraph:
    """Load and validate a graph generated by :func:`save_graph`."""

    np = _require_numpy()
    with np.load(path, allow_pickle=False) as values:
        node_count_value = values["node_count"] if "node_count" in values else None
        seed_value = values["seed"] if "seed" in values else None
        edge_source = tuple(int(value) for value in values["edge_source"].tolist())
        edge_target = tuple(int(value) for value in values["edge_target"].tolist())
        edge_sign = tuple(float(value) for value in values["edge_sign"].tolist())
        input_nodes = tuple(int(value) for value in values["input_nodes"].tolist())
        readout_nodes = tuple(int(value) for value in values["readout_nodes"].tolist())
        regions = tuple(str(value) for value in values["regions"].tolist())
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8")) if metadata_path else {}
    if not isinstance(metadata, dict):
        raise FlyNetFormatError("FlyNet graph metadata must be a JSON object")
    if metadata and metadata.get("schema_version") not in (None, "flynet.graph/v1"):
        raise FlyNetFormatError("unsupported FlyNet graph metadata schema")
    node_count = int(metadata.get("nodes", node_count_value if node_count_value is not None else len(regions)))
    seed = int(metadata.get("seed", seed_value if seed_value is not None else 0))
    return FlyNetGraph(
        node_count=node_count,
        edge_source=edge_source,
        edge_target=edge_target,
        edge_sign=edge_sign,
        input_nodes=input_nodes,
        readout_nodes=readout_nodes,
        regions=regions,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class FlyNetForward:
    """Observable outputs from a FlyNet forward pass."""

    policy_logits: Any
    value: Any
    activity: tuple[Any, ...] = ()


def _build_flynet_module():
    torch, nn = _require_torch()

    class _FlyNetModule(nn.Module):
        def __init__(self, graph: FlyNetGraph, *, steps: int, readout_width: int) -> None:
            super().__init__()
            if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 32:
                raise FlyNetFormatError("FlyNet steps must be an integer between 1 and 32")
            if readout_width < 8:
                raise FlyNetFormatError("FlyNet readout width must be at least 8")
            self.graph = graph
            self.steps = steps
            self.register_buffer("input_nodes", torch.tensor(graph.input_nodes, dtype=torch.long), persistent=False)
            self.register_buffer("readout_nodes", torch.tensor(graph.readout_nodes, dtype=torch.long), persistent=False)
            self.register_buffer("edge_source", torch.tensor(graph.edge_source, dtype=torch.long), persistent=False)
            self.register_buffer("edge_target", torch.tensor(graph.edge_target, dtype=torch.long), persistent=False)
            self.register_buffer("edge_sign", torch.tensor(graph.edge_sign, dtype=torch.float32))
            self.encoder = nn.Sequential(
                nn.Linear(FLYNET_FEATURES, len(graph.input_nodes)),
                nn.LayerNorm(len(graph.input_nodes)),
                nn.Tanh(),
            )
            self.edge_gain = nn.Parameter(torch.empty(graph.edge_count))
            self.node_bias = nn.Parameter(torch.zeros(graph.node_count))
            self.node_scale = nn.Parameter(torch.ones(steps, graph.node_count))
            self.readout = nn.Sequential(
                nn.Linear(len(graph.readout_nodes), readout_width),
                nn.LayerNorm(readout_width),
                nn.GELU(),
            )
            self.policy_head = nn.Linear(readout_width, FLYNET_POLICY_ACTIONS)
            self.value_head = nn.Linear(readout_width, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.normal_(self.edge_gain, mean=0.0, std=0.04)
            nn.init.normal_(self.node_bias, mean=0.0, std=0.02)
            nn.init.constant_(self.node_scale, 0.5)

        def forward(self, features: Any, *, include_activity: bool = False) -> FlyNetForward:
            if features.ndim != 2 or features.shape[1] != FLYNET_FEATURES:
                raise FlyNetFormatError(f"FlyNet features must have shape [batch, {FLYNET_FEATURES}]")
            drive_input = self.encoder(features)
            drive = features.new_zeros((features.shape[0], self.graph.node_count))
            drive = drive.index_copy(1, self.input_nodes, drive_input)
            state = features.new_zeros((features.shape[0], self.graph.node_count))
            activity: list[Any] = []
            edge_weight = self.edge_sign.to(features) * self.edge_gain.to(features)
            target = self.edge_target.to(features.device).expand(features.shape[0], -1)
            for step in range(self.steps):
                messages = features.new_zeros(state.shape)
                edge_messages = state[:, self.edge_source.to(features.device)] * edge_weight
                messages.scatter_add_(1, target, edge_messages)
                pre = (messages + drive + self.node_bias.to(features)) * self.node_scale[step].to(features)
                state = 0.65 * state + 0.35 * torch.tanh(pre)
                if include_activity:
                    activity.append(state)
            latent = self.readout(state[:, self.readout_nodes.to(features.device)])
            return FlyNetForward(
                policy_logits=self.policy_head(latent),
                value=torch.tanh(self.value_head(latent)).squeeze(-1),
                activity=tuple(activity),
            )

    return _FlyNetModule, torch


class FlyNetModel:
    """Trainable independent FlyNet graph model."""

    def __init__(self, graph: FlyNetGraph, *, steps: int = FLYNET_DEFAULT_STEPS, readout_width: int = 256, device: str = "cpu") -> None:
        module_type, torch = _build_flynet_module()
        try:
            self.device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise FlyNetFormatError(f"invalid FlyNet device: {device!r}") from exc
        try:
            self.module = module_type(graph, steps=steps, readout_width=readout_width).to(self.device)
        except (TypeError, RuntimeError) as exc:
            raise FlyNetFormatError(f"FlyNet device is unavailable: {device!r}") from exc

    @property
    def graph(self) -> FlyNetGraph:
        return self.module.graph

    @property
    def steps(self) -> int:
        return self.module.steps

    def parameters(self):
        return self.module.parameters()

    def train(self, mode: bool = True) -> "FlyNetModel":
        self.module.train(mode)
        return self

    def eval(self) -> "FlyNetModel":
        self.module.eval()
        return self

    def forward(self, features: Any, *, include_activity: bool = False) -> FlyNetForward:
        torch, _nn = _require_torch()
        if not isinstance(features, torch.Tensor):
            features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        return self.module(features.to(device=self.device, dtype=torch.float32), include_activity=include_activity)

    def state_dict(self) -> Any:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Any) -> None:
        self.module.load_state_dict(state_dict)

    def save_safetensors(self, path: str | Path) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise FlyNetDependencyError("SafeTensors export requires 'safetensors'") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {key: value.detach().cpu().contiguous() for key, value in self.state_dict().items()}
        save_file(tensors, str(path))

    def load_safetensors(self, path: str | Path) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise FlyNetDependencyError("SafeTensors loading requires 'safetensors'") from exc
        self.load_state_dict(load_file(str(path), device=str(self.device)))


class FlyNetPolicy:
    """Legal-move policy wrapper with inspectable neural telemetry."""

    def __init__(self, model: FlyNetModel) -> None:
        self.model = model.eval()
        self._last_forward: FlyNetForward | None = None
        self._last_readout: DecisionReadout | None = None
        self._reward_trace = 0.0

    @property
    def last_forward(self) -> FlyNetForward | None:
        return self._last_forward

    @property
    def last_readout(self) -> DecisionReadout | None:
        return self._last_readout

    @property
    def win_probability(self) -> float:
        if self._last_forward is None:
            return 0.5
        value = float(self._last_forward.value.detach().cpu().item())
        return (value + 1.0) / 2.0

    def select_move(self, board: chess.Board, legal_moves: Sequence[chess.Move] | None = None) -> chess.Move:
        torch, _nn = _require_torch()
        moves = tuple(legal_moves) if legal_moves is not None else tuple(board.legal_moves)
        if not moves:
            raise FlyNetFormatError("cannot select a move from a terminal position")
        with torch.no_grad():
            features = torch.tensor([encode_flynet_board(board)], dtype=torch.float32, device=self.model.device)
            result = self.model.forward(features, include_activity=True)
        self._last_forward = result
        logits = result.policy_logits[0].detach().cpu()
        legal_indices = [(move, action_index(move)) for move in moves]
        ranked = sorted(legal_indices, key=lambda pair: (float(logits[pair[1]]), pair[0].uci()), reverse=True)
        scale = torch.softmax(torch.tensor([float(logits[index]) for _move, index in ranked]), dim=0)
        activity = result.activity[-1][0].detach().cpu() if result.activity else None
        candidates = tuple(
            MoveCandidate(
                uci=move.uci(),
                san=board.san(move),
                score=float(scale[position]),
            )
            for position, (move, _index) in enumerate(ranked[:5])
        )
        self._last_readout = DecisionReadout(
            policy=type(self).__name__,
            selected_uci=ranked[0][0].uci(),
            candidates=candidates,
            activity=(
                ("active_nodes", float((activity.abs() > 0.05).sum().item()) if activity is not None else 0.0),
                ("node_count", float(self.model.graph.node_count)),
                ("recurrent_steps", float(self.model.steps)),
                ("value", float(result.value.detach().cpu().item())),
                ("reward_trace", self._reward_trace),
            ),
        )
        return ranked[0][0]

    def observe_reward(self, reward: float) -> None:
        """Keep the game-loop reward contract without mutating model weights."""

        self._reward_trace = 0.9 * self._reward_trace + float(reward)


__all__ = [
    "ACTION_INDEX",
    "ACTION_SPACE",
    "FLYNET_DEFAULT_EDGES",
    "FLYNET_DEFAULT_INPUT_NODES",
    "FLYNET_DEFAULT_NODES",
    "FLYNET_DEFAULT_READOUT_NODES",
    "FLYNET_DEFAULT_STEPS",
    "FLYNET_FEATURES",
    "FLYNET_POLICY_ACTIONS",
    "FLYNET_REGIONS",
    "FLYNET_VALUE_BINS",
    "FlyNetDependencyError",
    "FlyNetFormatError",
    "FlyNetForward",
    "FlyNetGraph",
    "FlyNetModel",
    "FlyNetPolicy",
    "action_index",
    "build_action_space",
    "encode_flynet_board",
    "generate_flynet_graph",
    "load_graph",
    "save_graph",
]
