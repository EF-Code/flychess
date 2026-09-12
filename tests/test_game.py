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


def test_game_loop_publishes_live_progress_phases() -> None:
    class FirstLegalMoveEngine:
        def choose_move(self, board):
            return next(iter(board.legal_moves))

    events: list[tuple[str, str, int]] = []

    result = play_game(
        SurrogateFlyBrain(seed=3),
        FirstLegalMoveEngine(),  # type: ignore[arg-type]
        max_plies=2,
        on_progress=lambda snapshot, phase, message: events.append(
            (phase, message, len(snapshot.moves))
        ),
    )

    assert len(result.moves) == 2
    assert [phase for phase, _message, _plies in events] == [
        "thinking",
        "decision",
        "move",
        "thinking",
        "move",
    ]
    decision_event = events[1]
    assert "candidates" in decision_event[1]
    assert decision_event[2] == 0
