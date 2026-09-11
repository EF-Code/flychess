"""Chess adapter for the generic connectome backend.

``ConnectomeFlyBrain`` is the bridge between a graph simulation and the
chess-specific interfaces. It deliberately keeps the adapter's assumptions
visible: board planes are projected cyclically onto configured input ports,
and output activity is decoded through a stable move hash. This is a
reproducible engineering experiment, not a calibrated fly visual system or
biological motor decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import chess

from .brain import DecisionReadout, MoveCandidate, encode_board
from .connectome import Connectome, ConnectomeValidationError


class ConnectomePolicyError(ValueError):
    """Raised when a graph cannot be used as a chess policy."""


@dataclass
class ConnectomeFlyBrain:
    """Use a :class:`Connectome` instance as a deterministic chess policy.

    The graph state persists across positions, allowing future recurrent or
    plastic models to carry state between turns. The current generic graph
    exposes no biologically validated visual or motor semantics, so the
    sensory projection and move decoder are intentionally documented here.
    """

    graph: Connectome
    steps_per_position: int = 2
    stimulation_scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.graph, Connectome):
            raise ConnectomePolicyError("graph must be a Connectome")
        if isinstance(self.steps_per_position, bool) or not isinstance(self.steps_per_position, int):
            raise ConnectomePolicyError("steps_per_position must be an integer")
        if not 1 <= self.steps_per_position <= 100:
            raise ConnectomePolicyError("steps_per_position must be between 1 and 100")
        if not isinstance(self.stimulation_scale, (int, float)) or isinstance(self.stimulation_scale, bool):
            raise ConnectomePolicyError("stimulation_scale must be a finite number")
        if not math.isfinite(float(self.stimulation_scale)):
            raise ConnectomePolicyError("stimulation_scale must be a finite number")
        if not self.graph.inputs:
            raise ConnectomePolicyError("connectome must declare at least one input port")
        if not self.graph.outputs:
            raise ConnectomePolicyError("connectome must declare at least one output port")
        self._reward_trace = 0.0
        self._positions = 0
        self._last_readout: DecisionReadout | None = None

    @property
    def positions_seen(self) -> int:
        return self._positions

    @property
    def reward_trace(self) -> float:
        return self._reward_trace

    @property
    def last_readout(self) -> DecisionReadout | None:
        """Return the latest observable candidate-move readout."""

        return self._last_readout

    def reset(self) -> None:
        """Reset graph state and adapter counters for a fresh experiment."""

        self.graph.reset()
        self._reward_trace = 0.0
        self._positions = 0
        self._last_readout = None

    def _stimulus(self, board: chess.Board) -> dict[str, float]:
        sensors = encode_board(board)
        return {
            node_id: float(sensors[index % len(sensors)]) * float(self.stimulation_scale)
            for index, node_id in enumerate(self.graph.inputs)
        }

    def _move_score(self, board: chess.Board, move: chess.Move, outputs: tuple[float, ...]) -> tuple[float, str]:
        """Score one legal move using graph output plus tiny deterministic priors."""

        output_index = (move.from_square * 64 + move.to_square) % len(outputs)
        score = outputs[output_index]
        if board.is_capture(move):
            score += 0.02
        if board.gives_check(move):
            score += 0.01
        score += 0.000001 * (move.to_square + (move.from_square / 64.0))
        return score, move.uci()

    def select_move(
        self, board: chess.Board, legal_moves: Sequence[chess.Move] | None = None
    ) -> chess.Move:
        moves = tuple(legal_moves) if legal_moves is not None else tuple(board.legal_moves)
        if not moves:
            raise ConnectomePolicyError("cannot select a move from a terminal position")
        try:
            self.graph.run(self.steps_per_position, stimulus=self._stimulus(board))
            # Read spikes rather than membrane voltage: the LIF backend resets
            # a neuron after it fires, so post-step membrane values can be zero
            # even when the output event was the signal we need to decode.
            outputs = self.graph.readout_vector(source="spikes")
        except ConnectomeValidationError as exc:
            raise ConnectomePolicyError(str(exc)) from exc
        self._positions += 1
        ranked = sorted(moves, key=lambda move: self._move_score(board, move, outputs), reverse=True)
        activity = tuple(self.graph.node_activity.values())
        self._last_readout = DecisionReadout(
            policy=type(self).__name__,
            selected_uci=ranked[0].uci(),
            candidates=tuple(
                MoveCandidate(
                    uci=move.uci(),
                    san=board.san(move),
                    score=self._move_score(board, move, outputs)[0],
                )
                for move in ranked[:5]
            ),
            activity=(
                ("active_nodes", float(sum(value != 0.0 for value in activity))),
                ("node_count", float(len(activity))),
                ("output_spikes", float(sum(self.graph.readout_vector(source="spikes")))),
                ("peak", max(activity, default=0.0)),
            ),
        )
        return ranked[0]

    def observe_reward(self, reward: float) -> None:
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ConnectomePolicyError("reward must be a finite number")
        if not math.isfinite(float(reward)):
            raise ConnectomePolicyError("reward must be a finite number")
        self._reward_trace = 0.9 * self._reward_trace + float(reward)


__all__ = ["ConnectomeFlyBrain", "ConnectomePolicyError"]
