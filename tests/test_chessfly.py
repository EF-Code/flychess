from __future__ import annotations

import struct

import chess
import pytest

from flychess.chessfly import (
    ACTION_INDEX,
    ACTION_SPACE,
    ChessFlyGraph,
    ChessFlyModel,
    ChessFlyPolicy,
    ChessFlyWeights,
    build_action_space,
    canonical_fen,
    encode_chessfly_fen,
    mirror_uci,
)


def _tiny_artifacts() -> tuple[bytes, bytes]:
    node_count = 4
    source_row_ptr = [0, 2, 3, 3, 4]
    targets = [1, 2, 2, 3]
    signs = [1, -1, 1, -1]
    connectome = bytearray(b"CF00")
    connectome.extend(struct.pack("<II", node_count, len(targets)))
    connectome.extend(struct.pack("<" + "I" * len(source_row_ptr), *source_row_ptr))
    connectome.extend(struct.pack("<" + "I" * len(targets), *targets))
    connectome.extend(struct.pack("<" + "h" * len(signs), *signs))
    neurons = bytearray(struct.pack("<" + "f" * (node_count * 3), *([0.0] * (node_count * 3))))
    neurons.extend(bytes([2, 1, 0, 3]))
    return bytes(connectome), bytes(neurons)


def test_graph_parser_transposes_source_csr_and_extracts_ports() -> None:
    connectome, neurons = _tiny_artifacts()

    graph = ChessFlyGraph.from_bytes(connectome, neurons)

    assert graph.summary() == {"neurons": 4, "edges": 4, "inputs": 1, "readout": 2}
    assert graph.row_ptr.tolist() == [0, 0, 1, 3, 4]
    assert graph.col_idx.tolist() == [0, 0, 1, 3]
    assert graph.sign.tolist() == [1, -1, 1, -1]
    assert graph.inputs.tolist() == [0]
    assert graph.readout.tolist() == [2, 3]


def test_chessfly_encoding_and_action_space_match_public_dimensions() -> None:
    features = encode_chessfly_fen("8/8/8/8/8/8/p6P/R3K2R w Kq b6 0 1")

    assert len(features) == 780
    assert features[chess.square(0, 0) * 12 + 3] == 1.0  # white rook a1
    assert features[chess.square(0, 1) * 12 + 6] == 1.0  # black pawn a2
    assert features[768] == 1.0
    assert features[769] == 0.0
    assert features[770] == 0.0
    assert features[771] == 1.0
    assert features[772 + 1] == 1.0
    assert len(ACTION_SPACE) == 1968
    assert ACTION_SPACE == build_action_space()
    assert ACTION_INDEX["a7a8q"] < len(ACTION_SPACE)
    assert ACTION_INDEX["a2a1q"] < len(ACTION_SPACE)
    assert "a8a1q" not in ACTION_INDEX


def test_black_perspective_mirroring_preserves_files_and_flips_ranks() -> None:
    fen, mirrored = canonical_fen("8/8/8/8/8/8/1p6/4k2K b kq e3 0 1")

    assert mirrored is True
    assert fen.split(" ")[:4] == ["4K2k/1P6/8/8/8/8/8/8", "w", "KQ", "e6"]
    assert mirror_uci("e2e1q") == "e7e8q"


def test_model_and_policy_run_with_small_compatible_tensors() -> None:
    torch = pytest.importorskip("torch")
    connectome, neurons = _tiny_artifacts()
    graph = ChessFlyGraph.from_bytes(connectome, neurons)
    tensors = {
        "encoder.weight": torch.zeros((1, 780)),
        "encoder.bias": torch.zeros(1),
        "log_gain": torch.zeros(4),
        "scale": torch.ones((1, 4)),
        "shift": torch.zeros((1, 4)),
        "decoder.weight": torch.zeros((2, 2)),
        "decoder.bias": torch.zeros(2),
        "policy.weight": torch.zeros((1968, 2)),
        "policy.bias": torch.zeros(1968),
        "value.weight": torch.zeros((64, 2)),
        "value.bias": torch.zeros(64),
    }
    model = ChessFlyModel(graph, ChessFlyWeights.from_tensors(tensors, {"steps": "1", "hidden": "2", "alpha": "0.5"}))
    output = model.forward(torch.zeros(780), include_activity=True)

    assert tuple(output.policy_logits.shape) == (1, 1968)
    assert tuple(output.value_logits.shape) == (1, 64)
    assert len(output.activity) == 1
    assert tuple(output.activity[0].shape) == (1, 4)

    policy = ChessFlyPolicy(model)
    move = policy.select_move(chess.Board())
    assert move in chess.Board().legal_moves
    assert policy.last_readout is not None
    assert policy.last_readout.selected_uci == move.uci()
    assert policy.win_probability == pytest.approx(0.5)
