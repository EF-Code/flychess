"""Game orchestration between a fly policy and Stockfish."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from .brain import DecisionReadout, FlyBrain
from .engine import StockfishEngine


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    actor: str
    san: str
    uci: str
    fen_before: str
    fen_after: str


@dataclass(frozen=True)
class DecisionRecord:
    """One fly-turn policy readout captured before the move is played."""

    ply: int
    fen: str
    readout: DecisionReadout


@dataclass
class GameResult:
    board: chess.Board
    moves: list[MoveRecord] = field(default_factory=list)
    start_fen: str = chess.Board().fen()
    fly_color: chess.Color = chess.WHITE
    max_plies: int = 160
    decision_trace: list[DecisionRecord] = field(default_factory=list)

    @property
    def outcome(self) -> chess.Outcome | None:
        return self.board.outcome(claim_draw=True)

    @property
    def stopped_at_limit(self) -> bool:
        return len(self.moves) >= self.max_plies and self.outcome is None


ProgressCallback = Callable[[GameResult, str, str], None]


def play_game(
    fly: FlyBrain,
    stockfish: StockfishEngine,
    *,
    fly_color: chess.Color = chess.WHITE,
    board: chess.Board | None = None,
    max_plies: int = 160,
    on_progress: ProgressCallback | None = None,
) -> GameResult:
    """Play one bounded game and return the complete final board.

    If supplied, ``on_progress`` receives immutable-in-practice snapshots with
    a phase (``thinking``, ``decision``, or ``move``) and a human-readable
    message. The callback is intended for live telemetry and does not alter
    move selection.
    """

    if max_plies < 1:
        raise ValueError("max_plies must be at least 1")
    current = board.copy(stack=True) if board is not None else chess.Board()
    start_fen = board.fen() if board is not None else chess.Board().fen()
    records: list[MoveRecord] = []
    decisions: list[DecisionRecord] = []

    def publish_progress(phase: str, message: str) -> None:
        if on_progress is None:
            return
        on_progress(
            GameResult(
                board=current.copy(stack=True),
                moves=records.copy(),
                start_fen=start_fen,
                fly_color=fly_color,
                max_plies=max_plies,
                decision_trace=decisions.copy(),
            ),
            phase,
            message,
        )

    while not current.is_game_over(claim_draw=True) and len(records) < max_plies:
        is_fly_turn = current.turn == fly_color
        legal_moves = tuple(current.legal_moves)
        fen_before = current.fen()
        actor = "Fly" if is_fly_turn else "Stockfish"
        if is_fly_turn:
            publish_progress(
                "thinking",
                f"Fly is weighing {len(legal_moves)} legal candidates for ply {len(records) + 1}…",
            )
        else:
            publish_progress("thinking", "Stockfish is calculating its reply…")
        move = fly.select_move(current, legal_moves) if is_fly_turn else stockfish.choose_move(current)
        if move not in legal_moves:
            raise RuntimeError(f"policy returned an illegal move: {move.uci()}")
        if is_fly_turn:
            readout = getattr(fly, "last_readout", None)
            if isinstance(readout, DecisionReadout):
                decisions.append(
                    DecisionRecord(
                        ply=len(records) + 1,
                        fen=fen_before,
                        readout=readout,
                    )
                )
                publish_progress(
                    "decision",
                    f"Fly selected {current.san(move)} from {len(readout.candidates)} surfaced candidates.",
                )
        san = current.san(move)
        current.push(move)
        records.append(
            MoveRecord(
                ply=len(records) + 1,
                actor="fly" if is_fly_turn else "stockfish",
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=current.fen(),
            )
        )
        publish_progress("move", f"{actor} played {san}.")

    outcome = current.outcome(claim_draw=True)
    if outcome is None:
        fly.observe_reward(0.0)
    elif outcome.winner is None:
        fly.observe_reward(0.0)
    else:
        fly.observe_reward(1.0 if outcome.winner == fly_color else -1.0)
    return GameResult(
        board=current,
        moves=records,
        start_fen=start_fen,
        fly_color=fly_color,
        max_plies=max_plies,
        decision_trace=decisions,
    )
