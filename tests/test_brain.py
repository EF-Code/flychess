import chess

from flychess.brain import SurrogateFlyBrain, encode_board


def test_board_encoding_has_expected_visual_contract() -> None:
    encoded = encode_board(chess.Board())
    assert len(encoded) == 6 * 2 * 64 + 5
    assert all(value in (0.0, 1.0, -1.0) for value in encoded)


def test_surrogate_brain_returns_a_legal_move() -> None:
    board = chess.Board()
    brain = SurrogateFlyBrain(seed=3)
    move = brain.select_move(board)
    assert move in board.legal_moves


def test_surrogate_brain_is_deterministic_for_same_seed() -> None:
    board = chess.Board()
    assert SurrogateFlyBrain(seed=19).select_move(board) == SurrogateFlyBrain(seed=19).select_move(board)
