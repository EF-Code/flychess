"""Canonical, filtered connectome records independent of source schemas."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .manifest import DatasetManifest


class CanonicalizationError(ValueError):
    """Raised when source rows cannot be normalized safely."""


def _key(value: Any) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _pick(row: Mapping[str, Any], *aliases: str) -> Any:
    normalized = {_key(name): value for name, value in row.items()}
    for alias in aliases:
        if _key(alias) in normalized:
            return normalized[_key(alias)]
    return None


def _body_id(row: Mapping[str, Any], *, context: str) -> int:
    value = _pick(row, "bodyId", "body_id", "body", "segmentId", "segment_id")
    if isinstance(value, bool) or value is None:
        raise CanonicalizationError(f"{context}: missing body ID")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(f"{context}: body ID must be an integer") from exc
    if parsed < 1:
        raise CanonicalizationError(f"{context}: body ID must be positive")
    return parsed


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped or None


def _optional_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise CanonicalizationError(f"{context}: value must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalizationError(f"{context}: value must be an integer") from exc
    if parsed < 0:
        raise CanonicalizationError(f"{context}: value must not be negative")
    return parsed


def _optional_float(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanonicalizationError(f"{context}: value must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CanonicalizationError(f"{context}: value must be finite")
    return parsed


@dataclass(frozen=True, slots=True)
class NeuronRecord:
    """Source-independent identity and annotation for one neuron/body."""

    body_id: int
    cell_type: str | None = None
    neuron_class: str | None = None
    side: str | None = None
    primary_neuropil: str | None = None
    neurotransmitters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.body_id, bool) or not isinstance(self.body_id, int) or self.body_id < 1:
            raise CanonicalizationError("neuron body_id must be a positive integer")
        object.__setattr__(self, "neurotransmitters", tuple(sorted(self.neurotransmitters)))


@dataclass(frozen=True, slots=True)
class ConnectionRecord:
    """One directed pre-synaptic to post-synaptic connection."""

    pre_body_id: int
    post_body_id: int
    synapse_count: int | None = None
    source_weight: float | None = None
    sign: float | None = None
    primary_neuropil: str | None = None

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.pre_body_id, self.post_body_id)
        ):
            raise CanonicalizationError("connection body IDs must be positive integers")
        synapse_count = _optional_int(self.synapse_count, context="connection synapse_count")
        source_weight = _optional_float(self.source_weight, context="connection source_weight")
        sign = _optional_float(self.sign, context="connection sign")
        if synapse_count is None and source_weight is None:
            raise CanonicalizationError("connection needs synapse_count or source_weight")
        object.__setattr__(self, "synapse_count", synapse_count)
        object.__setattr__(self, "source_weight", source_weight)
        object.__setattr__(self, "sign", sign)

    @property
    def effective_weight(self) -> float:
        """Return a source value suitable as a relative initial weight."""

        if self.source_weight is not None:
            return float(self.source_weight)
        return float(self.synapse_count or 0)


@dataclass(frozen=True, slots=True)
class CanonicalConnectome:
    """A deterministic, usually filtered, connectome subgraph."""

    manifest: DatasetManifest
    neurons: tuple[NeuronRecord, ...]
    connections: tuple[ConnectionRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, DatasetManifest):
            raise CanonicalizationError("canonical graph needs a DatasetManifest")
        neurons = tuple(sorted(self.neurons, key=lambda neuron: neuron.body_id))
        connections = tuple(sorted(self.connections, key=lambda edge: (edge.pre_body_id, edge.post_body_id)))
        node_ids = {neuron.body_id for neuron in neurons}
        if len(node_ids) != len(neurons):
            raise CanonicalizationError("canonical graph contains duplicate neuron IDs")
        if any(edge.pre_body_id not in node_ids or edge.post_body_id not in node_ids for edge in connections):
            raise CanonicalizationError("canonical graph contains an edge to an unknown neuron")
        object.__setattr__(self, "neurons", neurons)
        object.__setattr__(self, "connections", connections)

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(neuron.body_id for neuron in self.neurons)

    @property
    def node_count(self) -> int:
        return len(self.neurons)

    @property
    def edge_count(self) -> int:
        return len(self.connections)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_id": self.manifest.dataset_id,
            "source": self.manifest.source,
            "version": self.manifest.version,
            "sex": self.manifest.sex,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "body_id_min": min(self.node_ids) if self.node_ids else None,
            "body_id_max": max(self.node_ids) if self.node_ids else None,
        }


def _neurotransmitter_values(row: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    ignored = {
        _key(name)
        for name in (
            "bodyId",
            "body_id",
            "body",
            "segmentId",
            "segment_id",
            "instance",
            "type",
            "class",
            "side",
        )
    }
    values: list[tuple[str, float]] = []
    for name, raw_value in row.items():
        if _key(name) in ignored or raw_value is None:
            continue
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            continue
        value = float(raw_value)
        if math.isfinite(value):
            values.append((str(name), value))
    return tuple(sorted(values))


def canonicalize_malecns_rows(
    *,
    manifest: DatasetManifest,
    annotations: Iterable[Mapping[str, Any]],
    connections: Iterable[Mapping[str, Any]],
    neurotransmitters: Iterable[Mapping[str, Any]] = (),
    body_ids: Iterable[int] | None = None,
) -> CanonicalConnectome:
    """Normalize MaleCNS-like rows into a deterministic filtered graph.

    ``body_ids`` is the preferred production path: it limits the graph before
    the large connectivity table is materialized. Missing annotations are
    represented as explicit placeholder neurons rather than silently dropped.
    """

    selected = None if body_ids is None else set(body_ids)
    if selected is not None:
        if any(isinstance(body_id, bool) or not isinstance(body_id, int) or body_id < 1 for body_id in selected):
            raise CanonicalizationError("body_ids must contain positive integers")

    annotations_by_id: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(annotations):
        if not isinstance(row, Mapping):
            raise CanonicalizationError(f"annotations[{index}] must be a mapping")
        body_id = _body_id(row, context=f"annotations[{index}]")
        if selected is not None and body_id not in selected:
            continue
        if body_id in annotations_by_id:
            raise CanonicalizationError(f"annotations contain duplicate body ID {body_id}")
        annotations_by_id[body_id] = {
            "cell_type": _optional_text(_pick(row, "type", "cell_type", "cellType")),
            "neuron_class": _optional_text(_pick(row, "class", "neuron_class", "neuronClass")),
            "side": _optional_text(_pick(row, "side", "hemisphere")),
            "primary_neuropil": _optional_text(_pick(row, "primary_neuropil", "primaryNeuropil", "neuropil")),
        }

    neurotransmitters_by_id: dict[int, tuple[tuple[str, float], ...]] = {}
    for index, row in enumerate(neurotransmitters):
        if not isinstance(row, Mapping):
            raise CanonicalizationError(f"neurotransmitters[{index}] must be a mapping")
        body_id = _body_id(row, context=f"neurotransmitters[{index}]")
        if selected is not None and body_id not in selected:
            continue
        if body_id in neurotransmitters_by_id:
            raise CanonicalizationError(f"neurotransmitters contain duplicate body ID {body_id}")
        neurotransmitters_by_id[body_id] = _neurotransmitter_values(row)

    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for index, row in enumerate(connections):
        if not isinstance(row, Mapping):
            raise CanonicalizationError(f"connections[{index}] must be a mapping")
        pre_value = _pick(row, "bodyId_pre", "body_pre", "pre_body_id", "pre", "source")
        post_value = _pick(row, "bodyId_post", "body_post", "post_body_id", "post", "target")
        pre = _body_id({"bodyId": pre_value}, context=f"connections[{index}].pre")
        post = _body_id({"bodyId": post_value}, context=f"connections[{index}].post")
        if selected is not None and (pre not in selected or post not in selected):
            continue
        synapses = _optional_int(
            _pick(row, "synapse_count", "syn_count", "synapses", "count"),
            context=f"connections[{index}].synapse_count",
        )
        weight = _optional_float(
            _pick(row, "weight", "connection_strength", "source_weight"),
            context=f"connections[{index}].source_weight",
        )
        if synapses is None and weight is None:
            raise CanonicalizationError(f"connections[{index}] needs synapse count or weight")
        key = (pre, post)
        previous = edges.get(key)
        if previous is None:
            edges[key] = {
                "synapse_count": synapses,
                "source_weight": weight,
                "sign": _optional_float(_pick(row, "sign", "synapse_sign"), context=f"connections[{index}].sign"),
                "primary_neuropil": _optional_text(_pick(row, "primary_post", "primary_neuropil", "neuropil")),
            }
        else:
            if synapses is not None:
                previous["synapse_count"] = (previous["synapse_count"] or 0) + synapses
            if weight is not None:
                previous["source_weight"] = (previous["source_weight"] or 0.0) + weight

    all_ids = set(annotations_by_id) | set(neurotransmitters_by_id)
    all_ids.update(body_id for edge in edges for body_id in edge)
    neurons = tuple(
        NeuronRecord(
            body_id=body_id,
            **annotations_by_id.get(body_id, {}),
            neurotransmitters=neurotransmitters_by_id.get(body_id, ()),
        )
        for body_id in sorted(all_ids)
    )
    connection_records = tuple(
        ConnectionRecord(pre_body_id=pre, post_body_id=post, **values)
        for (pre, post), values in sorted(edges.items())
    )
    return CanonicalConnectome(manifest=manifest, neurons=neurons, connections=connection_records)
