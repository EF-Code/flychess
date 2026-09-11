"""Stockfish UCI integration."""

from __future__ import annotations

import shutil
from pathlib import Path

import chess
import chess.engine


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

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "StockfishEngine":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
