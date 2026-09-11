"""Fly-brain policies and the first, dependency-light baseline.

The public connectome releases provide wiring, not a drop-in biological
simulator with calibrated dynamics and motor semantics.  This module keeps
that boundary explicit: ``SurrogateFlyBrain`` is a deterministic recurrent
baseline used to validate the chess and Stockfish interfaces.  A future
MaleCNS/FlyWire backend can implement the same ``select_move`` and
``observe_reward`` methods without changing the game loop.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import chess


PIECE_PLANES: tuple[chess.PieceType, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


class FlyBrain(Protocol):
    """Interface required by the chess experiment."""

    def select_move(
        self, board: chess.Board, legal_moves: Sequence[chess.Move] | None = None
    ) -> chess.Move:
        """Choose one legal move from the current board."""

    def observe_reward(self, reward: float) -> None:
        """Deliver an experiment-level reward signal."""


def encode_board(board: chess.Board) -> tuple[float, ...]:
    """Encode a board as visual-style piece planes plus global context.

    The first 768 values are twelve 8x8 planes: six piece types for White,
    then six for Black.  Five global values represent side to move, castling
    rights, and whether an en-passant target exists.  This is intentionally
    simple and inspectable; it is the sensory contract a real fly backend can
    replace with photoreceptor stimulation.
    """

    features: list[float] = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_PLANES:
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                features.append(
                    1.0 if piece is not None and piece.color == color and piece.piece_type == piece_type else 0.0
                )

    features.extend(
        (
            1.0 if board.turn == chess.WHITE else -1.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
        )
    )
    return tuple(features)


def _piece_value(piece_type: chess.PieceType | None) -> float:
    return {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.2,
        chess.BISHOP: 3.3,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0,
    }.get(piece_type, 0.0)


def _action_features(board: chess.Board, move: chess.Move) -> tuple[float, ...]:
    """Extract compact action features for the baseline decoder."""

    moving = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    file_delta = abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) / 7.0
    rank_delta = abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) / 7.0
    destination_center = 1.0 - (
        abs(chess.square_file(move.to_square) - 3.5) + abs(chess.square_rank(move.to_square) - 3.5)
    ) / 7.0
    gives_check = board.gives_check(move)
    is_castle = board.is_castling(move)
    is_promotion = move.promotion is not None

    return (
        _piece_value(moving.piece_type if moving else None) / 9.0,
        _piece_value(captured.piece_type if captured else None) / 9.0,
        1.0 if captured is not None or board.is_en_passant(move) else 0.0,
        1.0 if gives_check else 0.0,
        1.0 if is_castle else 0.0,
        1.0 if is_promotion else 0.0,
        file_delta,
        rank_delta,
        destination_center,
        chess.square_file(move.to_square) / 7.0,
        chess.square_rank(move.to_square) / 7.0,
        1.0 if moving is not None and moving.color == chess.WHITE else -1.0,
    )


@dataclass
class SurrogateFlyBrain:
    """A deterministic recurrent policy for end-to-end plumbing.

    This is not a biological reconstruction.  It has a small recurrent
    association state, fixed sparse-looking weights, and an action decoder.
    Its purpose is to make the first chess experiment executable before a
    large connectome dataset and calibrated neural dynamics are integrated.
    """

    seed: int = 17
    hidden_size: int = 32
    leak: float = 0.82

    def __post_init__(self) -> None:
        if self.hidden_size < 4:
            raise ValueError("hidden_size must be at least 4")
        if not 0.0 < self.leak < 1.0:
            raise ValueError("leak must be between 0 and 1")
        self._hidden = [0.0] * self.hidden_size
        self._reward_trace = 0.0
        self._step = 0

    def _weight(self, index: int) -> float:
        # Stable pseudo-random structure without a model file or RNG state.
        return math.sin((index + 1) * (self.seed + 0.61803398875)) * 0.35

    def _integrate(self, sensors: Iterable[float]) -> None:
        sensor_values = tuple(sensors)
        mean_sensor = sum(sensor_values) / max(1, len(sensor_values))
        for index, previous in enumerate(self._hidden):
            probe = sensor_values[(index * 23 + self._step * 7) % len(sensor_values)]
            recurrent = self._hidden[(index * 5 + 3) % self.hidden_size]
            drive = self._weight(index) * (probe + mean_sensor) + 0.22 * recurrent
            self._hidden[index] = math.tanh(self.leak * previous + (1.0 - self.leak) * drive)

    def select_move(
        self, board: chess.Board, legal_moves: Sequence[chess.Move] | None = None
    ) -> chess.Move:
        moves = tuple(legal_moves) if legal_moves is not None else tuple(board.legal_moves)
        if not moves:
            raise ValueError("cannot select a move from a terminal position")

        sensors = encode_board(board)
        self._integrate(sensors)
        self._step += 1

        def score(move: chess.Move) -> tuple[float, str]:
            action = _action_features(board, move)
            neural = sum(
                self._hidden[index % self.hidden_size] * value * self._weight(index + 41)
                for index, value in enumerate(action)
            )
            reflex = 0.8 * action[1] + 0.35 * action[3] + 0.18 * action[8]
            novelty = 0.03 * math.sin(self._step + move.from_square * 0.7 + move.to_square * 0.13)
            reward_bias = 0.04 * self._reward_trace * (action[1] - action[0])
            return neural + reflex + novelty + reward_bias, move.uci()

        return max(moves, key=score)

    def observe_reward(self, reward: float) -> None:
        """Keep a decaying trace for future decoder/plasticity experiments."""

        self._reward_trace = 0.9 * self._reward_trace + float(reward)
