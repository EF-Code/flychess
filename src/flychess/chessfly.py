"""A compatible runtime for the public ChessFly artifact.

The ChessFly model card publishes learned tensors but keeps the graph in the
companion Space.  This module implements the small, explicit bridge between
those artifacts and Flychess:

* the Space's ``connectome.bin`` CSR-on-source format is validated and
  transposed into a CSR-on-target graph;
* the ``neurons.bin`` group labels define input and readout ports;
* board encoding, black-to-move mirroring, promotion action indexing, and the
  five calibrated recurrent steps match the public Space worker;
* legal-move masking remains an environment operation, and all move choices
  are surfaced as observable telemetry.

This is an artifact-compatible engineering runtime, not a claim that the
published model is a biological MaleCNS model or that its decoder has natural
motor meaning in a fly.  The graph and model files must be acquired separately
and their checksums should be recorded by the caller.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from .brain import DecisionReadout, MoveCandidate


CHESSFLY_FEATURES = 780
CHESSFLY_POLICY_ACTIONS = 1_968
CHESSFLY_VALUE_BINS = 64
CHESSFLY_GROUPS = frozenset({0, 1, 2, 3})
EXPECTED_KEYS = frozenset(
    {
        "decoder.bias",
        "decoder.weight",
        "encoder.bias",
        "encoder.weight",
        "log_gain",
        "policy.bias",
        "policy.weight",
        "scale",
        "shift",
        "value.bias",
        "value.weight",
    }
)


class ChessFlyDependencyError(RuntimeError):
    """Raised when optional NumPy, PyTorch, or Safetensors support is absent."""


class ChessFlyFormatError(ValueError):
    """Raised when ChessFly graph or tensor artifacts are malformed."""


class ChessFlyInferenceError(RuntimeError):
    """Raised when a ChessFly forward pass cannot be executed safely."""


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ChessFlyDependencyError(
            "ChessFly graph loading requires optional dependency 'numpy'"
        ) from exc
    return np


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ChessFlyDependencyError(
            "ChessFly inference requires optional dependency 'torch'"
        ) from exc
    return torch


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _decompress(path: str | Path) -> bytes:
    file_path = Path(path)
    try:
        payload = file_path.read_bytes()
    except OSError as exc:
        raise ChessFlyFormatError(f"could not read ChessFly artifact {file_path}: {exc}") from exc
    if file_path.suffix == ".gz":
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise ChessFlyFormatError(f"could not decompress ChessFly artifact {file_path}") from exc
    return payload


def _toggle_piece_case(rank: str) -> str:
    return "".join(character.swapcase() if character.isalpha() else character for character in rank)


def mirror_fen(fen: str) -> str:
    """Mirror a black-to-move FEN exactly as the public ChessFly worker does."""

    fields = fen.split(" ")
    if len(fields) < 4:
        raise ChessFlyFormatError("FEN must contain at least four space-separated fields")
    placement, turn, castling, en_passant, *rest = fields
    if turn not in {"w", "b"}:
        raise ChessFlyFormatError("FEN active color must be 'w' or 'b'")
    mirrored_castling = "-" if castling == "-" else "".join(
        sorted(_toggle_piece_case(castling), key="KQkq".index)
    )
    mirrored_ep = "-" if en_passant == "-" else en_passant[0] + str(9 - int(en_passant[1]))
    mirrored = [
        "/".join(_toggle_piece_case(rank) for rank in reversed(placement.split("/"))),
        "b" if turn == "w" else "w",
        mirrored_castling,
        mirrored_ep,
        *rest,
    ]
    return " ".join(mirrored)


def canonical_fen(fen: str) -> tuple[str, bool]:
    """Return a white-perspective FEN and whether the input was black-to-move."""

    fields = fen.split(" ")
    if len(fields) < 2 or fields[1] not in {"w", "b"}:
        raise ChessFlyFormatError("FEN active color must be 'w' or 'b'")
    mirrored = fields[1] == "b"
    return (mirror_fen(fen), True) if mirrored else (fen, False)


def mirror_uci(uci: str) -> str:
    """Mirror a UCI move's ranks, preserving files and promotion suffix."""

    if len(uci) < 4:
        raise ChessFlyFormatError(f"invalid UCI move {uci!r}")
    try:
        return f"{uci[0]}{9 - int(uci[1])}{uci[2]}{9 - int(uci[3])}{uci[4:]}"
    except (TypeError, ValueError) as exc:
        raise ChessFlyFormatError(f"invalid UCI move {uci!r}") from exc


def encode_chessfly_fen(fen: str) -> tuple[float, ...]:
    """Encode a canonical white-perspective FEN into 780 model features.

    Features 0--767 are twelve piece planes packed square-major: white pawn,
    knight, bishop, rook, queen, king, then the six black planes.  Features
    768--771 are KQkq castling flags, and 772--779 are en-passant-file flags.
    The caller should canonicalize black-to-move positions first.
    """

    fields = fen.split(" ")
    if len(fields) < 4:
        raise ChessFlyFormatError("FEN must contain at least four fields")
    placement, turn, castling, en_passant = fields[:4]
    if turn != "w":
        raise ChessFlyFormatError("encode_chessfly_fen expects a white-perspective FEN")
    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ChessFlyFormatError("FEN must contain eight ranks")
    values = [0.0] * CHESSFLY_FEATURES
    channels = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
    for fen_rank, rank in enumerate(ranks):
        file_index = 0
        for character in rank:
            if character.isdigit():
                file_index += int(character)
                continue
            channel = channels.get(character.lower())
            if channel is None or file_index >= 8:
                raise ChessFlyFormatError(f"invalid FEN placement {placement!r}")
            color_offset = 0 if character.isupper() else 6
            square = (7 - fen_rank) * 8 + file_index
            values[square * 12 + color_offset + channel] = 1.0
            file_index += 1
        if file_index != 8:
            raise ChessFlyFormatError(f"invalid FEN rank {rank!r}")
    if castling != "-":
        for index, flag in enumerate("KQkq"):
            values[768 + index] = 1.0 if flag in castling else 0.0
    if en_passant != "-":
        if len(en_passant) != 2 or en_passant[0] not in "abcdefgh" or en_passant[1] not in "12345678":
            raise ChessFlyFormatError(f"invalid en-passant square {en_passant!r}")
        values[772 + "abcdefgh".index(en_passant[0])] = 1.0
    return tuple(values)


def encode_chessfly_board(board: chess.Board) -> tuple[float, ...]:
    """Encode a board after mirroring it into the model's white perspective."""

    fen, _ = canonical_fen(board.fen())
    return encode_chessfly_fen(fen)


def build_action_space() -> tuple[str, ...]:
    """Build the sorted 1,968-action UCI space used by the public worker."""

    files = "abcdefgh"
    ranks = "12345678"
    actions: set[str] = set()
    ray_directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    knight_directions = (
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    )

    def square(rank: int, file_index: int) -> str:
        return files[file_index] + ranks[rank]

    for rank in range(8):
        for file_index in range(8):
            for rank_delta, file_delta in ray_directions:
                next_rank, next_file = rank + rank_delta, file_index + file_delta
                while 0 <= next_rank < 8 and 0 <= next_file < 8:
                    actions.add(square(rank, file_index) + square(next_rank, next_file))
                    next_rank += rank_delta
                    next_file += file_delta
            for rank_delta, file_delta in knight_directions:
                next_rank, next_file = rank + rank_delta, file_index + file_delta
                if 0 <= next_rank < 8 and 0 <= next_file < 8:
                    actions.add(square(rank, file_index) + square(next_rank, next_file))
    for rank in (6, 1):
        for file_index in range(8):
            for file_delta in (-1, 0, 1):
                target_file = file_index + file_delta
                if 0 <= target_file < 8:
                    for promotion in "qrbn":
                        target_rank = 7 if rank == 6 else 0
                        actions.add(square(rank, file_index) + square(target_rank, target_file) + promotion)
    return tuple(sorted(actions))


@dataclass(frozen=True, slots=True)
class ChessFlyGraph:
    """Validated target-row CSR graph and neuron ports for ChessFly."""

    node_count: int
    row_ptr: Any
    col_idx: Any
    sign: Any
    groups: Any
    inputs: Any
    readout: Any
    positions: Any | None = None

    def __post_init__(self) -> None:
        if isinstance(self.node_count, bool) or not isinstance(self.node_count, int) or self.node_count < 1:
            raise ChessFlyFormatError("node_count must be a positive integer")
        if len(self.row_ptr) != self.node_count + 1:
            raise ChessFlyFormatError("row_ptr length must be node_count + 1")
        edge_count = len(self.col_idx)
        if len(self.sign) != edge_count:
            raise ChessFlyFormatError("sign length must equal col_idx length")
        if len(self.groups) != self.node_count:
            raise ChessFlyFormatError("groups length must equal node_count")
        if int(self.row_ptr[0]) != 0 or int(self.row_ptr[-1]) != edge_count:
            raise ChessFlyFormatError("row_ptr must start at zero and end at edge count")
        if any(int(self.row_ptr[index]) > int(self.row_ptr[index + 1]) for index in range(self.node_count)):
            raise ChessFlyFormatError("row_ptr must be monotonic")
        if any(int(index) < 0 or int(index) >= self.node_count for index in self.col_idx):
            raise ChessFlyFormatError("col_idx contains an out-of-range node")
        if any(int(group) not in CHESSFLY_GROUPS for group in self.groups):
            raise ChessFlyFormatError("neurons.bin contains an unknown group label")
        if len(self.inputs) == 0 or len(self.readout) == 0:
            raise ChessFlyFormatError("graph must expose non-empty input and readout ports")

    @property
    def edge_count(self) -> int:
        return len(self.col_idx)

    @classmethod
    def from_bytes(cls, connectome: bytes, neurons: bytes) -> "ChessFlyGraph":
        """Parse the public Space's raw ``connectome.bin`` and ``neurons.bin``."""

        np = _numpy()
        if len(connectome) < 12:
            raise ChessFlyFormatError("connectome.bin is shorter than its header")
        node_count, edge_count = struct.unpack_from("<II", connectome, 4)
        expected_connectome_bytes = 12 + 4 * (node_count + 1) + 6 * edge_count
        if len(connectome) != expected_connectome_bytes:
            raise ChessFlyFormatError(
                f"connectome.bin size mismatch: expected {expected_connectome_bytes}, got {len(connectome)}"
            )
        row_offset = 12
        target_offset = row_offset + 4 * (node_count + 1)
        sign_offset = target_offset + 4 * edge_count
        source_row_ptr = np.frombuffer(connectome, dtype="<u4", count=node_count + 1, offset=row_offset).copy()
        targets = np.frombuffer(connectome, dtype="<u4", count=edge_count, offset=target_offset).copy()
        signs = np.frombuffer(connectome, dtype="<i2", count=edge_count, offset=sign_offset).copy()
        if int(source_row_ptr[0]) != 0 or int(source_row_ptr[-1]) != edge_count:
            raise ChessFlyFormatError("connectome source row pointer has invalid bounds")
        if np.any(source_row_ptr[1:] < source_row_ptr[:-1]):
            raise ChessFlyFormatError("connectome source row pointer is not monotonic")
        if np.any(targets >= node_count):
            raise ChessFlyFormatError("connectome contains an out-of-range target node")

        # The published binary is CSR by presynaptic source.  The worker
        # transposes it once so each postsynaptic row can gather its inputs.
        sources = np.repeat(np.arange(node_count, dtype=np.uint32), np.diff(source_row_ptr))
        stable_order = np.argsort(targets, kind="stable")
        sorted_targets = targets[stable_order]
        col_idx = sources[stable_order].astype(np.int64, copy=False)
        sign = signs[stable_order].astype(np.int16, copy=False)
        counts = np.bincount(sorted_targets, minlength=node_count)
        row_ptr = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))

        expected_neuron_bytes = 13 * node_count
        if len(neurons) != expected_neuron_bytes:
            raise ChessFlyFormatError(
                f"neurons.bin size mismatch: expected {expected_neuron_bytes}, got {len(neurons)}"
            )
        positions = np.frombuffer(neurons, dtype="<f4", count=node_count * 3, offset=0).reshape(node_count, 3).copy()
        groups = np.frombuffer(neurons, dtype="u1", count=node_count, offset=node_count * 12).copy()
        inputs = np.flatnonzero(groups == 2).astype(np.int64, copy=False)
        readout = np.flatnonzero((groups != 1) & (groups != 2)).astype(np.int64, copy=False)
        return cls(
            node_count=node_count,
            row_ptr=row_ptr,
            col_idx=col_idx,
            sign=sign,
            groups=groups,
            inputs=inputs,
            readout=readout,
            positions=positions,
        )

    @classmethod
    def from_files(cls, connectome_path: str | Path, neurons_path: str | Path) -> "ChessFlyGraph":
        return cls.from_bytes(_decompress(connectome_path), _decompress(neurons_path))

    def summary(self) -> dict[str, int]:
        return {
            "neurons": self.node_count,
            "edges": self.edge_count,
            "inputs": len(self.inputs),
            "readout": len(self.readout),
        }

    def sparse_matrix(self, log_gain: Any, *, device: str = "cpu") -> Any:
        """Build the calibrated target-row sparse matrix ``W``."""

        torch = _torch()
        if tuple(log_gain.shape) != (self.edge_count,):
            raise ChessFlyFormatError(
                f"log_gain shape must be [{self.edge_count}], got {tuple(log_gain.shape)}"
            )
        row_ptr = torch.as_tensor(self.row_ptr, dtype=torch.int64, device=device)
        col_idx = torch.as_tensor(self.col_idx, dtype=torch.int64, device=device)
        signs = torch.as_tensor(self.sign, dtype=torch.float32, device=device)
        signs = torch.where(signs < 0, -torch.ones_like(signs), torch.ones_like(signs))
        values = signs * torch.exp(log_gain.to(device=device, dtype=torch.float32))
        return torch.sparse_csr_tensor(
            row_ptr,
            col_idx,
            values,
            size=(self.node_count, self.node_count),
            device=device,
        )


@dataclass(frozen=True, slots=True)
class ChessFlyWeights:
    """Validated tensors and metadata from ``flynet.safetensors``."""

    tensors: Mapping[str, Any]
    steps: int
    alpha: float
    hidden: int
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        missing = EXPECTED_KEYS - set(self.tensors)
        if missing:
            raise ChessFlyFormatError(f"ChessFly weights are missing tensors: {', '.join(sorted(missing))}")
        if self.steps < 1 or self.hidden < 1 or not 0.0 < self.alpha <= 1.0:
            raise ChessFlyFormatError("ChessFly metadata has invalid steps, hidden size, or alpha")
        shapes = {name: tuple(value.shape) for name, value in self.tensors.items()}
        if shapes["encoder.weight"][1] != CHESSFLY_FEATURES:
            raise ChessFlyFormatError("encoder.weight must accept 780 features")
        if shapes["encoder.weight"][0] != shapes["encoder.bias"][0]:
            raise ChessFlyFormatError("encoder weight and bias input dimensions disagree")
        if shapes["decoder.weight"][0] != self.hidden or shapes["decoder.bias"] != (self.hidden,):
            raise ChessFlyFormatError("decoder dimensions disagree with hidden metadata")
        if shapes["policy.weight"] != (CHESSFLY_POLICY_ACTIONS, self.hidden):
            raise ChessFlyFormatError("policy.weight must have shape [1968, hidden]")
        if shapes["policy.bias"] != (CHESSFLY_POLICY_ACTIONS,):
            raise ChessFlyFormatError("policy.bias must have shape [1968]")
        if shapes["value.weight"] != (CHESSFLY_VALUE_BINS, self.hidden):
            raise ChessFlyFormatError("value.weight must have shape [64, hidden]")
        if shapes["value.bias"] != (CHESSFLY_VALUE_BINS,):
            raise ChessFlyFormatError("value.bias must have shape [64]")
        if shapes["scale"] != shapes["shift"]:
            raise ChessFlyFormatError("scale and shift shapes must match")
        if shapes["scale"][0] != self.steps:
            raise ChessFlyFormatError("scale step dimension disagrees with metadata")

    @property
    def encoder_inputs(self) -> int:
        return int(self.tensors["encoder.weight"].shape[0])

    @property
    def readout_neurons(self) -> int:
        return int(self.tensors["decoder.weight"].shape[1])

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, Any], metadata: Mapping[str, str] | None = None) -> "ChessFlyWeights":
        tensors = dict(tensors)
        metadata = dict(metadata or {})
        try:
            steps = int(metadata.get("steps", tensors["scale"].shape[0]))
            alpha = float(metadata.get("alpha", "0.5"))
            hidden = int(metadata.get("hidden", tensors["decoder.weight"].shape[0]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ChessFlyFormatError("ChessFly tensors or metadata are missing model dimensions") from exc
        return cls(tensors=tensors, steps=steps, alpha=alpha, hidden=hidden, metadata=metadata)

    @classmethod
    def from_safetensors(cls, path: str | Path, *, device: str = "cpu") -> "ChessFlyWeights":
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ChessFlyDependencyError(
                "ChessFly weights require optional dependency 'safetensors'"
            ) from exc
        try:
            tensors = load_file(str(path), device=device)
            with safe_open(str(path), framework="pt", device=device) as handle:
                metadata = handle.metadata() or {}
        except (OSError, RuntimeError, ValueError) as exc:
            raise ChessFlyFormatError(f"could not load ChessFly weights {path}: {exc}") from exc
        return cls.from_tensors(tensors, metadata)


@dataclass(frozen=True, slots=True)
class ChessFlyForward:
    """Outputs from one stateless batched ChessFly forward pass."""

    policy_logits: Any
    value_logits: Any
    activity: tuple[Any, ...] = ()


class ChessFlyModel:
    """Run the public five-step ChessFly dynamics over a validated graph."""

    def __init__(self, graph: ChessFlyGraph, weights: ChessFlyWeights, *, device: str = "cpu") -> None:
        torch = _torch()
        if graph.node_count != int(weights.tensors["scale"].shape[1]):
            raise ChessFlyFormatError("graph neuron count disagrees with scale tensor")
        if len(graph.inputs) != weights.encoder_inputs:
            raise ChessFlyFormatError("graph input-port count disagrees with encoder")
        if len(graph.readout) != weights.readout_neurons:
            raise ChessFlyFormatError("graph readout-port count disagrees with decoder")
        self.graph = graph
        self.weights = weights
        self.device = device
        self._tensors = {name: value.to(device=device, dtype=torch.float32) for name, value in weights.tensors.items()}
        self._inputs = torch.as_tensor(graph.inputs, dtype=torch.int64, device=device)
        self._readout = torch.as_tensor(graph.readout, dtype=torch.int64, device=device)
        self._matrix = graph.sparse_matrix(self._tensors["log_gain"], device=device)

    def forward(self, features: Any, *, include_activity: bool = False) -> ChessFlyForward:
        """Run one or more 780-feature canonical positions without autograd."""

        torch = _torch()
        with torch.inference_mode():
            return self._forward_impl(features, include_activity=include_activity)

    def _forward_impl(self, features: Any, *, include_activity: bool = False) -> ChessFlyForward:
        torch = _torch()
        functional = torch.nn.functional
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[1] != CHESSFLY_FEATURES:
            raise ChessFlyInferenceError("features must have shape [batch, 780]")
        tensors = self._tensors
        encoded = functional.linear(features, tensors["encoder.weight"], tensors["encoder.bias"])
        batch_size = features.shape[0]
        drive = torch.zeros((batch_size, self.graph.node_count), dtype=torch.float32, device=self.device)
        drive.index_copy_(1, self._inputs, encoded)
        hidden = torch.zeros_like(drive)
        activity: list[Any] = []
        for step in range(self.weights.steps):
            recurrent = torch.sparse.mm(self._matrix, hidden.transpose(0, 1)).transpose(0, 1)
            pre_activation = (recurrent + drive) * tensors["scale"][step] + tensors["shift"][step]
            hidden = (1.0 - self.weights.alpha) * hidden + self.weights.alpha * torch.relu(pre_activation)
            if include_activity:
                activity.append(hidden)
        readout = hidden.index_select(1, self._readout)
        association = functional.gelu(
            functional.linear(readout, tensors["decoder.weight"], tensors["decoder.bias"]),
            approximate="tanh",
        )
        return ChessFlyForward(
            policy_logits=functional.linear(association, tensors["policy.weight"], tensors["policy.bias"]),
            value_logits=functional.linear(association, tensors["value.weight"], tensors["value.bias"]),
            activity=tuple(activity),
        )

    @classmethod
    def from_artifacts(
        cls,
        connectome_path: str | Path,
        neurons_path: str | Path,
        weights_path: str | Path,
        *,
        device: str = "cpu",
    ) -> "ChessFlyModel":
        return cls(
            ChessFlyGraph.from_files(connectome_path, neurons_path),
            ChessFlyWeights.from_safetensors(weights_path, device=device),
            device=device,
        )


ACTION_SPACE = build_action_space()
ACTION_INDEX = {uci: index for index, uci in enumerate(ACTION_SPACE)}


class ChessFlyPolicy:
    """Expose :class:`ChessFlyModel` through Flychess's policy interface."""

    def __init__(self, model: ChessFlyModel) -> None:
        if not isinstance(model, ChessFlyModel):
            raise TypeError("model must be a ChessFlyModel")
        if len(ACTION_SPACE) != CHESSFLY_POLICY_ACTIONS:
            raise ChessFlyFormatError("ChessFly action-space construction did not produce 1968 actions")
        self.model = model
        self._positions = 0
        self._reward_trace = 0.0
        self._last_readout: DecisionReadout | None = None
        self._last_forward: ChessFlyForward | None = None
        self._win_probability: float | None = None

    @property
    def positions_seen(self) -> int:
        return self._positions

    @property
    def reward_trace(self) -> float:
        return self._reward_trace

    @property
    def last_readout(self) -> DecisionReadout | None:
        return self._last_readout

    @property
    def last_forward(self) -> ChessFlyForward | None:
        return self._last_forward

    @property
    def win_probability(self) -> float | None:
        return self._win_probability

    def reset(self) -> None:
        """Reset telemetry counters; each model forward pass is stateless."""

        self._positions = 0
        self._reward_trace = 0.0
        self._last_readout = None
        self._last_forward = None
        self._win_probability = None

    def select_move(
        self, board: chess.Board, legal_moves: Sequence[chess.Move] | None = None
    ) -> chess.Move:
        moves = tuple(legal_moves) if legal_moves is not None else tuple(board.legal_moves)
        if not moves:
            raise ChessFlyInferenceError("cannot select a move from a terminal position")
        fen, mirrored = canonical_fen(board.fen())
        canonical_legal = [mirror_uci(move.uci()) if mirrored else move.uci() for move in moves]
        indices = []
        for uci in canonical_legal:
            try:
                indices.append(ACTION_INDEX[uci])
            except KeyError as exc:
                raise ChessFlyInferenceError(f"legal move {uci!r} is outside ChessFly's action space") from exc
        torch = _torch()
        result = self.model.forward(torch.tensor(encode_chessfly_fen(fen)), include_activity=True)
        self._last_forward = result
        logits = result.policy_logits[0]
        selected_logits = logits[torch.as_tensor(indices, dtype=torch.int64, device=logits.device)]
        probabilities = torch.softmax(selected_logits, dim=0)
        ranked = sorted(
            zip(moves, probabilities.detach().cpu().tolist(), canonical_legal),
            key=lambda item: (-float(item[1]), item[2]),
        )
        selected = ranked[0][0]
        value_probabilities = torch.softmax(result.value_logits[0], dim=0)
        self._win_probability = float(
            sum(float(probability) * ((index + 0.5) / CHESSFLY_VALUE_BINS) for index, probability in enumerate(value_probabilities))
        )
        final_activity = result.activity[-1][0].detach().cpu() if result.activity else None
        activity = [
            ("node_count", float(self.model.graph.node_count)),
            ("input_count", float(len(self.model.graph.inputs))),
            ("readout_count", float(len(self.model.graph.readout))),
            ("legal_candidates", float(len(moves))),
            ("win_probability", self._win_probability),
        ]
        if final_activity is not None:
            activity.extend(
                (
                    ("active_nodes", float((final_activity > 0).sum().item())),
                    ("mean_activity", float(final_activity.mean().item())),
                    ("peak_activity", float(final_activity.max().item())),
                )
            )
        self._positions += 1
        self._last_readout = DecisionReadout(
            policy=type(self).__name__,
            selected_uci=selected.uci(),
            candidates=tuple(
                MoveCandidate(uci=move.uci(), san=board.san(move), score=float(probability))
                for move, probability, _ in ranked[:5]
            ),
            activity=tuple(activity),
        )
        return selected

    def observe_reward(self, reward: float) -> None:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ChessFlyInferenceError("reward must be a finite number")
        self._reward_trace = 0.9 * self._reward_trace + float(reward)


def sha256_file(path: str | Path) -> str:
    """Hash an acquired artifact for a reproducibility manifest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ACTION_INDEX",
    "ACTION_SPACE",
    "CHESSFLY_FEATURES",
    "CHESSFLY_POLICY_ACTIONS",
    "CHESSFLY_VALUE_BINS",
    "ChessFlyDependencyError",
    "ChessFlyFormatError",
    "ChessFlyForward",
    "ChessFlyGraph",
    "ChessFlyInferenceError",
    "ChessFlyModel",
    "ChessFlyPolicy",
    "ChessFlyWeights",
    "build_action_space",
    "canonical_fen",
    "encode_chessfly_board",
    "encode_chessfly_fen",
    "mirror_fen",
    "mirror_uci",
    "sha256_file",
]
