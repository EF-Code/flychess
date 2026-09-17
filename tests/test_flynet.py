from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest

from flychess.flynet import (
    ACTION_INDEX,
    ACTION_SPACE,
    FLYNET_FEATURES,
    FlyNetFormatError,
    FlyNetGraph,
    encode_flynet_board,
    generate_flynet_graph,
    load_graph,
    save_graph,
)
from flychess.flynet_training import FlyNetDataset, load_dataset, save_dataset


def test_flynet_encoder_and_action_space_are_independent_contracts() -> None:
    board = chess.Board()
    encoded = encode_flynet_board(board)

    assert len(encoded) == FLYNET_FEATURES
    assert encoded[768] == 1.0
    assert len(ACTION_SPACE) == 4544
    assert ACTION_SPACE == tuple(sorted(ACTION_INDEX))
    assert ACTION_INDEX["e2e4"] < len(ACTION_SPACE)
    assert ACTION_INDEX["a7a8q"] < len(ACTION_SPACE)


def test_graph_generation_is_seeded_and_round_trips(tmp_path: Path) -> None:
    graph = generate_flynet_graph(
        node_count=48,
        edge_count=256,
        input_nodes=8,
        readout_nodes=15,
        seed=91,
    )
    same = generate_flynet_graph(
        node_count=48,
        edge_count=256,
        input_nodes=8,
        readout_nodes=15,
        seed=91,
    )
    assert graph == same
    assert graph.summary()["regions"]["optic"] == 8
    assert graph.regions[-15:] == ("action",) * 7 + ("value",) * 8

    graph_path = tmp_path / "flynet-graph.npz"
    metadata_path = tmp_path / "metadata" / "graph.json"
    save_graph(graph, graph_path, metadata_path)
    assert load_graph(graph_path, metadata_path) == graph
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["schema_version"] == "flynet.graph/v1"


def test_graph_rejects_impossible_dimensions() -> None:
    with pytest.raises(FlyNetFormatError, match="capacity"):
        generate_flynet_graph(node_count=8, edge_count=57, input_nodes=2, readout_nodes=2)
    with pytest.raises(FlyNetFormatError, match="non-port"):
        generate_flynet_graph(node_count=8, edge_count=8, input_nodes=4, readout_nodes=4)


def test_dataset_round_trip_and_checksum_validation(tmp_path: Path) -> None:
    fens = (chess.Board().fen(), chess.Board().fen())
    dataset = FlyNetDataset(
        features=np.zeros((2, FLYNET_FEATURES), dtype=np.float32),
        action_indices=np.array([ACTION_INDEX["e2e4"], ACTION_INDEX["d2d4"]], dtype=np.int64),
        values=np.array([0.0, 0.25], dtype=np.float32),
        fens=fens,
        metadata={"seed": 9},
    )
    data_path = tmp_path / "dataset.npz"
    save_dataset(dataset, data_path)
    loaded = load_dataset(data_path)
    assert loaded.sample_count == 2
    assert np.array_equal(loaded.action_indices, dataset.action_indices)

    data_path.write_bytes(data_path.read_bytes() + b"tamper")
    with pytest.raises(FlyNetFormatError, match="checksum"):
        load_dataset(data_path)


def test_small_model_forward_backward_and_legal_policy() -> None:
    torch = pytest.importorskip("torch")
    from flychess.flynet import FlyNetModel, FlyNetPolicy

    graph = generate_flynet_graph(
        node_count=32,
        edge_count=128,
        input_nodes=8,
        readout_nodes=8,
        seed=4,
    )
    model = FlyNetModel(graph, steps=2, readout_width=16)
    output = model.forward(torch.zeros((2, FLYNET_FEATURES), dtype=torch.float32), include_activity=True)
    assert tuple(output.policy_logits.shape) == (2, 4544)
    assert tuple(output.value.shape) == (2,)
    assert len(output.activity) == 2
    loss = output.policy_logits[:, :8].square().mean() + output.value.square().mean()
    loss.backward()
    assert model.module.edge_gain.grad is not None

    policy = FlyNetPolicy(model)
    board = chess.Board()
    move = policy.select_move(board)
    assert move in board.legal_moves
    assert policy.last_readout is not None
    assert policy.last_readout.selected_uci == move.uci()
    policy.observe_reward(1.0)
