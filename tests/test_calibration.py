from __future__ import annotations

import json

import pytest

from flychess.calibration import (
    Artifact,
    DatasetManifest,
    LIFParameters,
    ManifestVerificationError,
    SparseLIF,
    canonicalize_malecns_rows,
    sha256_file,
)
from flychess.calibration.canonical import CanonicalizationError


def _manifest(*artifacts: Artifact) -> DatasetManifest:
    return DatasetManifest(
        source="male-cns",
        version="v1.0",
        sex="male",
        dataset_id="male-cns:v1.0",
        license="CC-BY",
        artifacts=artifacts,
    )


def test_manifest_round_trips_and_verifies_hashed_artifacts(tmp_path) -> None:
    artifact_path = tmp_path / "annotations.feather"
    artifact_path.write_bytes(b"fixture")
    digest = sha256_file(artifact_path)
    manifest = _manifest(Artifact("annotations", "annotations.feather", digest))

    restored = DatasetManifest.from_mapping(json.loads(json.dumps(manifest.as_dict())))
    assert restored == manifest
    assert restored.is_locked
    restored.verify(tmp_path)


def test_manifest_rejects_hash_mismatch_and_path_escape(tmp_path) -> None:
    (tmp_path / "artifact.bin").write_bytes(b"actual")
    manifest = _manifest(Artifact("artifact", "artifact.bin", "0" * 64))
    with pytest.raises(ManifestVerificationError, match="hash mismatch"):
        manifest.verify(tmp_path)
    with pytest.raises(ValueError, match="must be relative"):
        Artifact("bad", "../outside.bin")


def test_canonicalizer_filters_rows_and_preserves_annotations() -> None:
    manifest = _manifest(Artifact("annotations", "annotations.feather"))
    graph = canonicalize_malecns_rows(
        manifest=manifest,
        annotations=[
            {"bodyId": 10, "type": "DNge104", "class": "descending", "side": "L", "neuropil": "VNC"},
            {"bodyId": 20, "type": "motor", "class": "motor", "side": "R", "neuropil": "VNC"},
            {"bodyId": 30, "type": "excluded", "class": "other"},
        ],
        connections=[
            {"bodyId_pre": 10, "bodyId_post": 20, "weight": 2.5, "sign": 1.0},
            {"bodyId_pre": 10, "bodyId_post": 20, "synapse_count": 3},
            {"bodyId_pre": 20, "bodyId_post": 30, "synapse_count": 7},
        ],
        neurotransmitters=[{"bodyId": 10, "GABA": 0.1, "acetylcholine": 0.9}],
        body_ids=[10, 20],
    )

    assert graph.node_ids == (10, 20)
    assert graph.node_count == 2
    assert graph.edge_count == 1
    assert graph.connections[0].synapse_count == 3
    assert graph.connections[0].source_weight == 2.5
    assert graph.neurons[0].cell_type == "DNge104"
    assert graph.neurons[0].neurotransmitters == (("GABA", 0.1), ("acetylcholine", 0.9))


def test_canonicalizer_requires_a_connection_weight() -> None:
    manifest = _manifest(Artifact("annotations", "annotations.feather"))
    with pytest.raises(CanonicalizationError, match="synapse count or weight"):
        canonicalize_malecns_rows(
            manifest=manifest,
            annotations=[],
            connections=[{"bodyId_pre": 1, "bodyId_post": 2}],
        )


def test_canonicalizer_rejects_duplicate_selected_annotations() -> None:
    manifest = _manifest(Artifact("annotations", "annotations.feather"))
    with pytest.raises(CanonicalizationError, match="duplicate body ID 1"):
        canonicalize_malecns_rows(
            manifest=manifest,
            annotations=[{"bodyId": 1}, {"bodyId": 1}],
            connections=[],
        )


def test_sparse_lif_uses_previous_spikes_and_explicit_synapse_signs() -> None:
    manifest = _manifest(Artifact("connectivity", "connectivity.feather"))
    graph = canonicalize_malecns_rows(
        manifest=manifest,
        annotations=[{"bodyId": 1}, {"bodyId": 2}, {"bodyId": 3}],
        connections=[
            {"bodyId_pre": 1, "bodyId_post": 2, "synapse_count": 4, "sign": 1.0},
            {"bodyId_pre": 1, "bodyId_post": 3, "synapse_count": 4, "sign": -1.0},
        ],
    )
    dynamics = SparseLIF(
        graph,
        inputs=[1],
        outputs=[2, 3],
        node_parameters={
            body_id: LIFParameters(leak=0.0)
            for body_id in graph.node_ids
        },
        weight_scale=0.5,
        synapse_exponent=0.5,
    )

    first = dynamics.step({1: 1.0})
    second = dynamics.step()

    assert first.activity == {1: 1.0, 2: 0.0, 3: 0.0}
    assert second.activity == {1: 0.0, 2: 1.0, 3: 0.0}
    assert second.membrane[3] == -1.0
    assert dynamics.readout_vector() == (1.0, 0.0)
    assert dynamics.effective_edges == ((1, 2, 1.0), (1, 3, -1.0))


def test_sparse_lif_rejects_invalid_ports_and_bounds_runs() -> None:
    manifest = _manifest(Artifact("connectivity", "connectivity.feather"))
    graph = canonicalize_malecns_rows(
        manifest=manifest,
        annotations=[{"bodyId": 1}],
        connections=[],
    )
    with pytest.raises(ValueError, match="unknown body ID"):
        SparseLIF(graph, inputs=[99])

    dynamics = SparseLIF(graph, max_steps=1)
    assert dynamics.run(0) == ()
    with pytest.raises(ValueError, match="at most"):
        dynamics.run(2)
