import chess
import pytest

from flychess.connectome import Connectome
from flychess.connectome_policy import ConnectomeFlyBrain, ConnectomePolicyError


def _graph() -> Connectome:
    output_ids = [f"out-{index:03d}" for index in range(16)]
    return Connectome(
        [{"id": "sensor", "bias": 1.0}, *output_ids],
        [["sensor", output_id, 1.0] for output_id in output_ids],
        inputs=["sensor"],
        outputs=output_ids,
    )


def test_connectome_policy_stimulates_graph_and_returns_legal_move() -> None:
    graph = _graph()
    brain = ConnectomeFlyBrain(graph, steps_per_position=2)
    board = chess.Board()

    move = brain.select_move(board)

    assert move in board.legal_moves
    assert graph.steps_run == 2
    assert brain.positions_seen == 1
    assert any(value == 1.0 for value in graph.readout().values())
    assert brain.last_readout is not None
    assert brain.last_readout.selected_uci == move.uci()
    assert len(brain.last_readout.candidates) == 5
    assert dict(brain.last_readout.activity)["node_count"] == len(graph.node_ids)


def test_connectome_policy_preserves_state_until_explicit_reset() -> None:
    brain = ConnectomeFlyBrain(_graph())
    board = chess.Board()
    brain.select_move(board)
    brain.observe_reward(1.0)
    assert brain.positions_seen == 1
    assert brain.reward_trace == 1.0

    brain.reset()

    assert brain.positions_seen == 0
    assert brain.reward_trace == 0.0
    assert brain.graph.steps_run == 0


def test_connectome_policy_requires_ports_and_valid_options() -> None:
    with pytest.raises(ConnectomePolicyError, match="input port"):
        ConnectomeFlyBrain(
            Connectome(
                ["only-output"],
                [["only-output", "only-output", 1.0]],
                outputs=["only-output"],
            )
        )
    with pytest.raises(ConnectomePolicyError, match="output port"):
        ConnectomeFlyBrain(
            Connectome(
                ["only-input"],
                [["only-input", "only-input", 1.0]],
                inputs=["only-input"],
            )
        )
    with pytest.raises(ConnectomePolicyError, match="between"):
        ConnectomeFlyBrain(_graph(), steps_per_position=0)
