import shutil

import chess
import pytest

from flychess.engine import StockfishEngine


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish is not installed")
def test_stockfish_returns_a_legal_move() -> None:
    board = chess.Board()
    with StockfishEngine(depth=1) as engine:
        move = engine.choose_move(board)
    assert move in board.legal_moves

