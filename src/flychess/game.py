"""Game orchestration between a fly policy and Stockfish."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .brain import FlyBrain
from .engine import StockfishEngine


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    actor: str
    san: str
    uci: str
    fen_before: str
    fen_after: str


@dataclass
class GameResult:
    board: chess.Board
    moves: list[MoveRecord] = field(default_factory=list)
    start_fen: str = chess.Board().fen()
    fly_color: chess.Color = chess.WHITE
    max_plies: int = 160

    @property
    def outcome(self) -> chess.Outcome | None:
        return self.board.outcome(claim_draw=True)

    @property
    def stopped_at_limit(self) -> bool:
        return len(self.moves) >= self.max_plies and self.outcome is None


def play_game(
    fly: FlyBrain,
    stockfish: StockfishEngine,
    *,
    fly_color: chess.Color = chess.WHITE,
    board: chess.Board | None = None,
    max_plies: int = 160,
) -> GameResult:
    """Play one bounded game and return the complete final board."""

    if max_plies < 1:
        raise ValueError("max_plies must be at least 1")
    current = board.copy(stack=True) if board is not None else chess.Board()
    records: list[MoveRecord] = []

    while not current.is_game_over(claim_draw=True) and len(records) < max_plies:
        is_fly_turn = current.turn == fly_color
        legal_moves = tuple(current.legal_moves)
        move = fly.select_move(current, legal_moves) if is_fly_turn else stockfish.choose_move(current)
        if move not in legal_moves:
            raise RuntimeError(f"policy returned an illegal move: {move.uci()}")
        fen_before = current.fen()
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
        start_fen=board.fen() if board is not None else chess.Board().fen(),
        fly_color=fly_color,
        max_plies=max_plies,
    )
