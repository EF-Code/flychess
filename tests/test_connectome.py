import json

import pytest

from flychess.connectome import (
    Connectome,
    ConnectomeFormatError,
    ConnectomeValidationError,
    load_connectome,
)


def _document(*, reverse: bool = False) -> dict[str, object]:
    nodes = [
        {"id": "sensor", "threshold": 1.0, "leak": 0.5},
        {"id": "relay", "threshold": 0.75, "leak": 0.25},
        {"id": "out", "threshold": 1.0, "leak": 0.0},
    ]
    edges = [
        ["sensor", "relay", 1.0],
        {"source": "relay", "target": "out", "weight": 1.25},
    ]
    if reverse:
        nodes.reverse()
        edges.reverse()
    return {
        "nodes": nodes,
        "edges": edges,
        "inputs": ["sensor"],
        "outputs": ["out"],
    }


def test_loads_documented_edge_list_from_path_and_exposes_ports(tmp_path) -> None:
    path = tmp_path / "tiny-connectome.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    graph = Connectome.from_json(path)

    assert graph.node_ids == ("out", "relay", "sensor")
    assert graph.inputs == ("sensor",)
    assert graph.outputs == ("out",)
    assert [(edge.source, edge.target, edge.weight) for edge in graph.edges] == [
        ("relay", "out", 1.25),
        ("sensor", "relay", 1.0),
    ]
    assert dict(graph.node_activity) == {"out": 0.0, "relay": 0.0, "sensor": 0.0}


def test_loads_json_text_and_empty_graph_with_derived_ports() -> None:
    document = {"nodes": ["source", "sink"], "edges": [["source", "sink", 0.5]]}

    graph = load_connectome(json.dumps(document))

    assert graph.inputs == ("source",)
    assert graph.outputs == ("sink",)
    assert graph.readout() == {"sink": 0.0}
    assert graph.run(0) == ()

    empty = Connectome.from_json({"nodes": [], "edges": []})
    assert empty.node_ids == ()
    assert empty.node_activity == {}
    assert empty.readout_vector() == ()
    assert empty.step().activity == {}


@pytest.mark.parametrize(
    "document",
    [
        {"edges": []},
        {"nodes": []},
        {"nodes": "sensor", "edges": []},
        {"nodes": ["sensor"], "edges": [["sensor", "missing", 1.0]]},
        {"nodes": ["sensor"], "edges": [["sensor", "sensor"]]},
        {"nodes": [{"threshold": 1.0}], "edges": []},
        {"nodes": ["sensor"], "edges": [{"source": "sensor", "target": "sensor"}]},
    ],
)
def test_rejects_malformed_documents(document) -> None:
    with pytest.raises((ConnectomeFormatError, ConnectomeValidationError, ValueError)):
        Connectome.from_json(document)


def test_rejects_invalid_json_and_bad_node_edge_values() -> None:
    with pytest.raises(ConnectomeFormatError, match="invalid connectome JSON"):
        Connectome.from_json("{not-json")

    with pytest.raises(ConnectomeFormatError, match="finite"):
        Connectome.from_json(
            {
                "nodes": ["a", "b"],
                "edges": [["a", "b", float("nan")]],
            }
        )
    with pytest.raises(ConnectomeValidationError, match="duplicate"):
        Connectome(["a"], [["a", "a", 1.0], ["a", "a", 2.0]])
    with pytest.raises(ConnectomeValidationError, match="unknown node"):
        Connectome(["a"], [["a", "missing", 1.0]])


def test_step_uses_previous_spikes_and_lif_reset() -> None:
    graph = Connectome.from_json(
        {
            "nodes": [
                {"id": "sensor", "threshold": 1.0, "leak": 0.0},
                {"id": "out", "threshold": 1.0, "leak": 0.0},
            ],
            "edges": [["sensor", "out", 1.0]],
            "inputs": ["sensor"],
            "outputs": ["out"],
        }
    )

    first = graph.step({"sensor": 1.0})
    second = graph.step()

    assert first.activity == {"out": 0.0, "sensor": 1.0}
    assert second.activity == {"out": 1.0, "sensor": 0.0}
    assert first.membrane["sensor"] == 0.0
    assert second.membrane["out"] == 0.0
    assert graph.readout() == {"out": 1.0}
    assert graph.readout(source="membrane") == {"out": 0.0}
    assert graph.readout_vector() == (1.0,)


def test_stimulation_queues_once_and_accumulates_with_saturation() -> None:
    graph = Connectome(
        [{"id": "sensor", "threshold": 1.0, "leak": 0.0}],
        inputs=["sensor"],
        outputs=["sensor"],
        potential_limit=2.0,
    )
    graph.stimulate("sensor", 0.6)
    graph.stimulate({"sensor": 0.6})

    first = graph.step()
    second = graph.step()

    assert first.activity["sensor"] == 1.0
    assert second.activity["sensor"] == 0.0
    assert second.membrane["sensor"] == 0.0

    with pytest.raises(ConnectomeValidationError, match="unknown node"):
        graph.stimulate({"missing": 1.0})


def test_run_is_bounded_and_rejects_ambiguous_or_invalid_requests() -> None:
    graph = Connectome(["sensor"], max_steps=2)

    assert len(graph.run(2, stimulus={"sensor": 0.1})) == 2
    assert graph.steps_run == 2
    with pytest.raises(ConnectomeValidationError, match="at most"):
        graph.run(3)
    with pytest.raises(ConnectomeValidationError, match="at least"):
        graph.run(-1)
    with pytest.raises(ConnectomeValidationError, match="either"):
        graph.run(1, stimulus={"sensor": 0.1}, stimulation={"sensor": 0.1})
    with pytest.raises(ConnectomeValidationError, match="source"):
        graph.readout(source="unknown")


def test_activity_snapshots_are_read_only_and_reset_isolated() -> None:
    graph = Connectome(["sensor"], outputs=["sensor"])
    snapshot = graph.node_activity
    with pytest.raises(TypeError):
        snapshot["sensor"] = 1.0

    graph.step({"sensor": 1.0})
    assert snapshot["sensor"] == 0.0
    graph.reset()
    assert graph.steps_run == 0
    assert graph.node_activity == {"sensor": 0.0}
    assert graph.membrane_potentials == {"sensor": 0.0}


def test_equivalent_node_and_edge_order_has_identical_activity() -> None:
    first = Connectome.from_json(_document())
    second = Connectome.from_json(_document(reverse=True))
    stimuli = [{"sensor": 1.0}, None, None, {"sensor": 0.25}]

    first_results = tuple(first.step(stimulus) for stimulus in stimuli)
    second_results = tuple(second.step(stimulus) for stimulus in stimuli)

    assert [result.activity for result in first_results] == [result.activity for result in second_results]
    assert [result.membrane for result in first_results] == [result.membrane for result in second_results]
    assert first.readout() == second.readout()


def test_refractory_period_suppresses_immediate_repeated_spikes() -> None:
    graph = Connectome(
        [{"id": "sensor", "threshold": 1.0, "leak": 0.0, "refractory_steps": 1}],
        outputs=["sensor"],
    )

    results = graph.run(3, stimulus={"sensor": 1.0})

    assert [result.activity["sensor"] for result in results] == [1.0, 0.0, 1.0]


def test_potentials_remain_bounded_for_large_finite_drive() -> None:
    graph = Connectome(
        [{"id": "sensor", "threshold": 2.0, "leak": 0.99, "bias": 1e300}],
        potential_limit=3.0,
    )

    results = graph.run(5, stimulus={"sensor": -1e300})

    assert all(abs(result.membrane["sensor"]) <= 3.0 for result in results)
