import shutil

import chess
import pytest

from flychess.brain import SurrogateFlyBrain
from flychess.engine import StockfishEngine
from flychess.game import play_game


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish is not installed")
def test_game_loop_keeps_both_players_legal() -> None:
    with StockfishEngine(depth=1) as engine:
        result = play_game(SurrogateFlyBrain(), engine, max_plies=6)
    assert len(result.moves) == 6
    assert result.board.fullmove_number >= 4
    assert all(record.actor in {"fly", "stockfish"} for record in result.moves)
    assert result.moves[0].fen_before == chess.Board().fen()
    assert result.moves[-1].fen_after == result.board.fen()
    assert result.stopped_at_limit
    assert result.decision_trace
    assert all(decision.readout.selected_uci for decision in result.decision_trace)
    assert all(
        decision.fen == move.fen_before
        for decision, move in zip(result.decision_trace, result.moves[::2])
    )


def test_game_loop_rejects_invalid_bound() -> None:
    class NeverUsed:
        def select_move(self, board, legal_moves=None):
            raise AssertionError

        def observe_reward(self, reward):
            raise AssertionError

    with pytest.raises(ValueError, match="max_plies"):
        play_game(NeverUsed(), None, max_plies=0)  # type: ignore[arg-type]
