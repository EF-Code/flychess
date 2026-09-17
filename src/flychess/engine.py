"""Stockfish UCI integration."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine


@dataclass(frozen=True, slots=True)
class EngineAssessment:
    """A reproducible Stockfish label for one position."""

    move: chess.Move
    score_cp: int | None
    value: float


def _score_to_value(score: Any) -> tuple[int | None, float]:
    """Map a python-chess POV score to a bounded side-to-move value."""

    if score.is_mate():
        mate_distance = score.mate()
        if mate_distance is None:
            return None, 0.0
        return None, 1.0 if mate_distance > 0 else -1.0
    score_cp = score.score(mate_score=100_000)
    if score_cp is None:
        return None, 0.0
    return int(score_cp), math.tanh(float(score_cp) / 400.0)


class StockfishEngine:
    """Small lifecycle-safe wrapper around a Stockfish process."""

    def __init__(self, path: str = "stockfish", depth: int = 4) -> None:
        if depth < 1:
            raise ValueError("depth must be at least 1")
        resolved = str(Path(path).expanduser()) if "/" in path or path.startswith(".") else shutil.which(path)
        if not resolved:
            raise FileNotFoundError(
                f"Stockfish executable not found: {path!r}. Install Stockfish or pass --engine PATH."
            )
        self.path = resolved
        self.depth = depth
        self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
        try:
            self._engine.configure({"Threads": 1, "Hash": 16})
        except chess.engine.EngineError:
            # Some packaged builds expose a smaller UCI option set.
            pass

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot choose a move from a terminal position")
        result = self._engine.play(board, chess.engine.Limit(depth=self.depth))
        if result.move is None or result.move not in board.legal_moves:
            raise RuntimeError("Stockfish returned no legal move")
        return result.move

    def assess(self, board: chess.Board) -> EngineAssessment:
        """Return Stockfish's best move and bounded value for ``board``.

        The value is always from the side-to-move perspective.  This makes it
        suitable as a supervised target for FlyNet and avoids accidentally
        mixing White- and Black-perspective labels during dataset generation.
        """

        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot assess a terminal position")
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
        move = info.get("pv", [None])[0]
        if move is None:
            move = self.choose_move(board)
        if move not in board.legal_moves:
            raise RuntimeError("Stockfish returned an illegal principal-variation move")
        score = info.get("score")
        if score is None:
            return EngineAssessment(move=move, score_cp=None, value=0.0)
        score_cp, value = _score_to_value(score.pov(board.turn))
        return EngineAssessment(move=move, score_cp=score_cp, value=value)

    def top_moves(self, board: chess.Board, *, count: int = 5) -> tuple[chess.Move, ...]:
        """Return up to ``count`` principal-variation moves in score order."""

        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot rank moves from a terminal position")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 32:
            raise ValueError("count must be an integer between 1 and 32")
        infos = self._engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=count,
        )
        if isinstance(infos, dict):
            infos = [infos]
        moves: list[chess.Move] = []
        for info in infos:
            pv = info.get("pv", [])
            move = pv[0] if pv else None
            if move is not None and move in board.legal_moves and move not in moves:
                moves.append(move)
        return tuple(moves)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
