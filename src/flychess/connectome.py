"""A small deterministic connectome-style graph backend.

The backend intentionally uses a compact, documented edge-list format rather
than depending on a particular connectomics data release.  A JSON document has
this shape::

    {
      "nodes": [
        "sensor",
        {"id": "readout", "threshold": 0.5, "leak": 0.8}
      ],
      "edges": [["sensor", "readout", 1.25]],
      "inputs": ["sensor"],
      "outputs": ["readout"]
    }

``nodes`` must be a list of unique non-empty string IDs, or node objects with
an ``id`` and optional ``threshold``, ``leak``, ``reset``, ``bias``, and
``refractory_steps`` fields.  ``edges`` must contain three-item
``[source, target, weight]`` records or equivalent objects with
``source``, ``target``, and ``weight`` fields.  Endpoints must be declared
nodes; directed duplicate edges are rejected.  Self-edges are allowed and use
the previous step's spike.  ``inputs`` and ``outputs`` are optional ordered
lists of node IDs.  If omitted, they default to source nodes (no incoming
edges) and sink nodes (no outgoing edges), respectively.

``Connectome.from_json`` accepts a filesystem path, JSON text, bytes, or an
already-decoded mapping.  ``stimulate`` queues a one-step sensory current;
``step`` advances one synchronous leaky-integrate-and-fire-style update; and
``run`` advances a caller-bounded number of steps.  Membrane potentials are
saturated at ``potential_limit`` so malformed or unusually large inputs cannot
make the numerical state diverge.  Node IDs are sorted internally, and edge
contribution order is canonicalized, so equivalent JSON documents produce the
same deterministic activity.

This is an engineering backend for experiments.  It is not a reconstruction
of a living fly, a calibrated biological connectome simulator, or evidence of
biological equivalence, learning, consciousness, or motor behavior.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias


DEFAULT_MAX_STEPS = 1_000
DEFAULT_POTENTIAL_LIMIT = 10.0
MAX_ALLOWED_STEPS = 1_000_000

JsonObject: TypeAlias = Mapping[str, Any]
JsonSource: TypeAlias = str | bytes | bytearray | Path | Mapping[str, Any]


class ConnectomeError(ValueError):
    """Base class for malformed graph documents and invalid simulation input."""


class ConnectomeFormatError(ConnectomeError):
    """Raised when a JSON document does not have the supported shape."""


class ConnectomeValidationError(ConnectomeError):
    """Raised when graph nodes, edges, or runtime configuration are invalid."""


def _finite_number(value: Any, label: str) -> float:
    """Return a finite float, rejecting booleans and numeric strings."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConnectomeValidationError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ConnectomeValidationError(f"{label} must be a finite number")
    return converted


def _non_empty_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConnectomeValidationError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConnectomeValidationError(f"{label} must be an integer")
    if value < minimum:
        raise ConnectomeValidationError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConnectomeValidationError(f"{label} must be at most {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Validated parameters for one simulated node.

    ``leak`` is the fraction of the previous distance from ``reset`` retained
    on each step.  A value of ``0`` forgets the previous membrane potential;
    values closer to ``1`` retain more of it.  A node fires when its candidate
    potential reaches ``threshold`` and is then set to ``reset``.
    """

    id: str
    threshold: float = 1.0
    leak: float = 0.8
    reset: float = 0.0
    bias: float = 0.0
    refractory_steps: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _non_empty_id(self.id, "node id"))
        threshold = _finite_number(self.threshold, f"node {self.id!r} threshold")
        if threshold <= 0.0:
            raise ConnectomeValidationError(f"node {self.id!r} threshold must be greater than zero")
        leak = _finite_number(self.leak, f"node {self.id!r} leak")
        if not 0.0 <= leak < 1.0:
            raise ConnectomeValidationError(f"node {self.id!r} leak must be in [0, 1)")
        reset = _finite_number(self.reset, f"node {self.id!r} reset")
        bias = _finite_number(self.bias, f"node {self.id!r} bias")
        refractory = _integer(
            self.refractory_steps,
            f"node {self.id!r} refractory_steps",
            minimum=0,
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "leak", leak)
        object.__setattr__(self, "reset", reset)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "refractory_steps", refractory)


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """A validated directed weighted connection between two nodes."""

    source: str
    target: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _non_empty_id(self.source, "edge source"))
        object.__setattr__(self, "target", _non_empty_id(self.target, "edge target"))
        object.__setattr__(self, "weight", _finite_number(self.weight, "edge weight"))


@dataclass(frozen=True, slots=True)
class StepResult:
    """Immutable state produced by one synchronous simulation step.

    ``activity`` and ``spikes`` are equivalent named views containing ``0.0``
    or ``1.0`` for every node.  ``membrane`` contains the post-step bounded
    membrane potentials.  All mappings are read-only snapshots.
    """

    step_index: int
    activity: Mapping[str, float]
    membrane: Mapping[str, float]
    spikes: Mapping[str, float]

    def __getitem__(self, node_id: str) -> float:
        """Allow ``result[node_id]`` as a short form for spike activity."""

        return self.activity[node_id]

    @property
    def membrane_potentials(self) -> Mapping[str, float]:
        """Alias for the post-step membrane snapshot."""

        return self.membrane


# Short aliases make the data records convenient without duplicating types.
Node = NodeSpec
Edge = EdgeSpec


def _read_only(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(values))


def _as_items(value: Any, label: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise ConnectomeFormatError(f"{label} must be a list")
    return list(value)


def _parse_node(value: Any, index: int) -> NodeSpec:
    label = f"nodes[{index}]"
    if isinstance(value, str):
        return NodeSpec(value)
    if not isinstance(value, Mapping):
        raise ConnectomeFormatError(f"{label} must be a string or object")
    if "id" not in value:
        raise ConnectomeFormatError(f"{label} is missing 'id'")
    allowed = {"id", "threshold", "leak", "reset", "bias", "refractory_steps"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConnectomeFormatError(f"{label} has unknown fields: {', '.join(map(str, unknown))}")
    try:
        return NodeSpec(
            id=value["id"],
            threshold=value.get("threshold", 1.0),
            leak=value.get("leak", 0.8),
            reset=value.get("reset", 0.0),
            bias=value.get("bias", 0.0),
            refractory_steps=value.get("refractory_steps", 0),
        )
    except ConnectomeValidationError as exc:
        raise ConnectomeFormatError(f"{label}: {exc}") from exc


def _parse_edge(value: Any, index: int) -> EdgeSpec:
    label = f"edges[{index}]"
    if isinstance(value, Mapping):
        required = {"source", "target", "weight"}
        missing = sorted(required - set(value))
        if missing:
            raise ConnectomeFormatError(f"{label} is missing: {', '.join(missing)}")
        unknown = sorted(set(value) - required)
        if unknown:
            raise ConnectomeFormatError(f"{label} has unknown fields: {', '.join(map(str, unknown))}")
        source, target, weight = value["source"], value["target"], value["weight"]
    else:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise ConnectomeFormatError(f"{label} must be a three-item list or object")
        if len(value) != 3:
            raise ConnectomeFormatError(f"{label} must contain source, target, and weight")
        source, target, weight = value
    try:
        return EdgeSpec(source=source, target=target, weight=weight)
    except ConnectomeValidationError as exc:
        raise ConnectomeFormatError(f"{label}: {exc}") from exc


def _parse_ports(value: Any, label: str, node_ids: set[str]) -> tuple[str, ...]:
    items = _as_items(value, label)
    ports: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        try:
            node_id = _non_empty_id(item, f"{label}[{index}]")
        except ConnectomeValidationError as exc:
            raise ConnectomeFormatError(str(exc)) from exc
        if node_id not in node_ids:
            raise ConnectomeValidationError(f"{label}[{index}] references unknown node {node_id!r}")
        if node_id in seen:
            raise ConnectomeValidationError(f"{label} contains duplicate node {node_id!r}")
        seen.add(node_id)
        ports.append(node_id)
    return tuple(ports)


def _load_document(source: JsonSource) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        document = source
    else:
        if isinstance(source, Path):
            try:
                text = source.read_text(encoding="utf-8")
            except OSError as exc:
                raise ConnectomeFormatError(f"could not read connectome JSON {source}: {exc}") from exc
        elif isinstance(source, (bytes, bytearray)):
            try:
                text = bytes(source).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ConnectomeFormatError("connectome JSON bytes must be UTF-8") from exc
        elif isinstance(source, str):
            stripped = source.lstrip()
            if stripped.startswith(("{", "[")):
                text = source
            else:
                path = Path(source)
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError as exc:
                    raise ConnectomeFormatError(f"could not read connectome JSON {path}: {exc}") from exc
        else:
            raise TypeError("connectome source must be a path, JSON text, bytes, or mapping")
        try:
            document = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ConnectomeFormatError(f"invalid connectome JSON: {exc}") from exc

    if not isinstance(document, Mapping):
        raise ConnectomeFormatError("connectome JSON root must be an object")
    return document


class Connectome:
    """A bounded deterministic synchronous LIF-style graph simulator.

    The graph is immutable after construction; simulation state belongs to the
    instance and can be cleared with :meth:`reset`.  A call to :meth:`run` is
    limited to ``max_steps`` steps.  ``stimulus`` passed to ``run`` is applied
    before every step, while :meth:`stimulate` queues a current for the next
    one only.
    """

    def __init__(
        self,
        nodes: Iterable[NodeSpec | Mapping[str, Any] | str],
        edges: Iterable[EdgeSpec | Mapping[str, Any] | Sequence[Any]] = (),
        *,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        potential_limit: float = DEFAULT_POTENTIAL_LIMIT,
    ) -> None:
        parsed_nodes = self._parse_nodes(nodes)
        node_ids = {node.id for node in parsed_nodes}
        if len(node_ids) != len(parsed_nodes):
            raise ConnectomeValidationError("node IDs must be unique")
        ordered_nodes = tuple(sorted(parsed_nodes, key=lambda node: node.id))

        parsed_edges = self._parse_edges(edges, node_ids)
        edge_pairs = {(edge.source, edge.target) for edge in parsed_edges}
        if len(edge_pairs) != len(parsed_edges):
            raise ConnectomeValidationError("directed duplicate edges are not allowed")

        validated_limit = _finite_number(potential_limit, "potential_limit")
        if validated_limit <= 0.0:
            raise ConnectomeValidationError("potential_limit must be greater than zero")
        validated_max_steps = _integer(
            max_steps,
            "max_steps",
            minimum=1,
            maximum=MAX_ALLOWED_STEPS,
        )
        for node in ordered_nodes:
            if abs(node.reset) > validated_limit:
                raise ConnectomeValidationError(
                    f"node {node.id!r} reset must be within potential_limit"
                )
            if node.threshold > validated_limit:
                raise ConnectomeValidationError(
                    f"node {node.id!r} threshold must not exceed potential_limit"
                )

        incoming = {node.id: [] for node in ordered_nodes}
        outgoing = {node.id: [] for node in ordered_nodes}
        for edge in parsed_edges:
            incoming[edge.target].append((edge.source, edge.weight))
            outgoing[edge.source].append(edge.target)
        self._incoming = {
            node_id: tuple(sorted(connections, key=lambda connection: (connection[0], connection[1])))
            for node_id, connections in incoming.items()
        }
        self._outgoing = {
            node_id: tuple(sorted(targets)) for node_id, targets in outgoing.items()
        }

        if inputs is None:
            input_ports = tuple(
                node_id for node_id in self._node_ids_from(ordered_nodes) if not self._incoming[node_id]
            )
        else:
            input_ports = _parse_ports(inputs, "inputs", node_ids)
        if outputs is None:
            output_ports = tuple(
                node_id for node_id in self._node_ids_from(ordered_nodes) if not self._outgoing[node_id]
            )
        else:
            output_ports = _parse_ports(outputs, "outputs", node_ids)

        self._nodes = ordered_nodes
        self._node_map = {node.id: node for node in ordered_nodes}
        self._node_ids = tuple(node.id for node in ordered_nodes)
        self._inputs = input_ports
        self._outputs = output_ports
        self._edges = tuple(sorted(parsed_edges, key=lambda edge: (edge.source, edge.target, edge.weight)))
        self.max_steps = validated_max_steps
        self.potential_limit = validated_limit
        self._step_index = 0
        self._membrane: dict[str, float] = {}
        self._spikes: dict[str, float] = {}
        self._refractory: dict[str, int] = {}
        self._pending_stimulation: dict[str, float] = {}
        self.reset()

    @staticmethod
    def _node_ids_from(nodes: Iterable[NodeSpec]) -> tuple[str, ...]:
        return tuple(node.id for node in nodes)

    @staticmethod
    def _parse_nodes(nodes: Iterable[NodeSpec | Mapping[str, Any] | str]) -> tuple[NodeSpec, ...]:
        if isinstance(nodes, Mapping):
            # A small convenience for programmatic use: {"id": {options}}.
            values: list[Any] = []
            for node_id, options in nodes.items():
                if options is None:
                    values.append(node_id)
                elif isinstance(options, Mapping):
                    values.append({"id": node_id, **dict(options)})
                else:
                    raise ConnectomeValidationError(
                        f"node mapping entry {node_id!r} must contain an options object"
                    )
        else:
            if isinstance(nodes, (str, bytes, bytearray)) or not isinstance(nodes, Iterable):
                raise ConnectomeValidationError("nodes must be an iterable of node records")
            values = list(nodes)
        parsed: list[NodeSpec] = []
        for index, value in enumerate(values):
            if isinstance(value, NodeSpec):
                parsed.append(value)
            else:
                try:
                    parsed.append(_parse_node(value, index))
                except ConnectomeFormatError as exc:
                    raise ConnectomeValidationError(str(exc)) from exc
        return tuple(parsed)

    @staticmethod
    def _parse_edges(
        edges: Iterable[EdgeSpec | Mapping[str, Any] | Sequence[Any]], node_ids: set[str]
    ) -> tuple[EdgeSpec, ...]:
        if isinstance(edges, (str, bytes, bytearray)) or not isinstance(edges, Iterable):
            raise ConnectomeValidationError("edges must be an iterable of edge records")
        parsed: list[EdgeSpec] = []
        for index, value in enumerate(edges):
            if isinstance(value, EdgeSpec):
                edge = value
            else:
                try:
                    edge = _parse_edge(value, index)
                except ConnectomeFormatError as exc:
                    raise ConnectomeValidationError(str(exc)) from exc
            if edge.source not in node_ids or edge.target not in node_ids:
                missing = edge.source if edge.source not in node_ids else edge.target
                raise ConnectomeValidationError(
                    f"edges[{index}] references unknown node {missing!r}"
                )
            parsed.append(edge)
        return tuple(parsed)

    @classmethod
    def from_json(
        cls,
        source: JsonSource,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        potential_limit: float = DEFAULT_POTENTIAL_LIMIT,
    ) -> "Connectome":
        """Load and validate a connectome from the documented JSON format."""

        document = _load_document(source)
        for required in ("nodes", "edges"):
            if required not in document:
                raise ConnectomeFormatError(f"connectome JSON requires a '{required}' field")
        try:
            nodes = _as_items(document["nodes"], "nodes")
            edges = _as_items(document["edges"], "edges")
        except ConnectomeFormatError:
            raise
        inputs = document.get("inputs")
        outputs = document.get("outputs")
        if inputs is not None and (isinstance(inputs, (str, bytes, bytearray)) or not isinstance(inputs, Iterable)):
            raise ConnectomeFormatError("inputs must be a list")
        if outputs is not None and (isinstance(outputs, (str, bytes, bytearray)) or not isinstance(outputs, Iterable)):
            raise ConnectomeFormatError("outputs must be a list")
        try:
            return cls(
                nodes,
                edges,
                inputs=inputs,
                outputs=outputs,
                max_steps=max_steps,
                potential_limit=potential_limit,
            )
        except ConnectomeValidationError as exc:
            raise ConnectomeFormatError(str(exc)) from exc

    @classmethod
    def load(cls, source: JsonSource, **kwargs: Any) -> "Connectome":
        """Alias for :meth:`from_json` for callers that prefer ``load``."""

        return cls.from_json(source, **kwargs)

    @property
    def node_ids(self) -> tuple[str, ...]:
        """Canonical sorted node IDs."""

        return self._node_ids

    @property
    def inputs(self) -> tuple[str, ...]:
        """Ordered sensory port IDs."""

        return self._inputs

    @property
    def outputs(self) -> tuple[str, ...]:
        """Ordered readout port IDs."""

        return self._outputs

    @property
    def nodes(self) -> tuple[NodeSpec, ...]:
        """Immutable node specifications in canonical order."""

        return self._nodes

    @property
    def edges(self) -> tuple[EdgeSpec, ...]:
        """Immutable directed edge specifications in canonical order."""

        return self._edges

    @property
    def steps_run(self) -> int:
        """Number of steps since the last reset."""

        return self._step_index

    @property
    def node_activity(self) -> Mapping[str, float]:
        """Read-only latest spike activity for every node."""

        return _read_only(self._spikes)

    @property
    def activity(self) -> Mapping[str, float]:
        """Alias for :attr:`node_activity`."""

        return self.node_activity

    @property
    def spikes(self) -> Mapping[str, float]:
        """Read-only latest spike activity for every node."""

        return self.node_activity

    @property
    def membrane_potentials(self) -> Mapping[str, float]:
        """Read-only latest bounded membrane potential for every node."""

        return _read_only(self._membrane)

    def get_activity(self) -> Mapping[str, float]:
        """Return a read-only snapshot of latest node spike activity."""

        return self.node_activity

    def reset(self) -> None:
        """Reset membrane, spikes, refractory timers, pending input, and step count."""

        self._step_index = 0
        self._membrane = {node.id: node.reset for node in self._nodes}
        self._spikes = {node.id: 0.0 for node in self._nodes}
        self._refractory = {node.id: 0 for node in self._nodes}
        self._pending_stimulation = {node.id: 0.0 for node in self._nodes}

    def _normalise_stimulus(
        self,
        stimulus: Mapping[str, Any] | Iterable[tuple[str, Any]],
        *,
        label: str = "stimulus",
    ) -> dict[str, float]:
        if isinstance(stimulus, Mapping):
            items: Iterator[tuple[Any, Any]] = iter(stimulus.items())
        else:
            if isinstance(stimulus, (str, bytes, bytearray)) or not isinstance(stimulus, Iterable):
                raise ConnectomeValidationError(f"{label} must map node IDs to numeric currents")
            items = iter(stimulus)
        values: dict[str, float] = {}
        for index, item in enumerate(items):
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes, bytearray)) or len(item) != 2:
                raise ConnectomeValidationError(f"{label}[{index}] must be a (node_id, value) pair")
            node_id = _non_empty_id(item[0], f"{label}[{index}] node")
            if node_id not in self._node_map:
                raise ConnectomeValidationError(f"{label} references unknown node {node_id!r}")
            amount = _finite_number(item[1], f"{label}[{index}] value")
            values[node_id] = self._saturate(values.get(node_id, 0.0) + amount)
        return values

    def _saturate(self, value: float) -> float:
        if not math.isfinite(value):
            return math.copysign(self.potential_limit, value)
        return min(self.potential_limit, max(-self.potential_limit, value))

    def stimulate(
        self,
        stimulus: Mapping[str, Any] | Iterable[tuple[str, Any]] | str,
        value: Any | None = None,
    ) -> None:
        """Queue sensory current for the next step.

        The mapping form is ``stimulate({"sensor": 1.0})``.  The shorthand
        ``stimulate("sensor", 1.0)`` is equivalent.  Multiple calls before a
        step add currents with saturation at ``potential_limit``.
        """

        if isinstance(stimulus, str):
            if value is None:
                raise ConnectomeValidationError("stimulate(node_id, value) requires a value")
            values = self._normalise_stimulus(((stimulus, value),))
        else:
            if value is not None:
                raise ConnectomeValidationError("value is only valid with a single node ID")
            values = self._normalise_stimulus(stimulus)
        for node_id, amount in values.items():
            self._pending_stimulation[node_id] = self._saturate(
                self._pending_stimulation[node_id] + amount
            )

    def _step_stimulus(self, stimulus: Mapping[str, Any] | Iterable[tuple[str, Any]] | None) -> dict[str, float]:
        if stimulus is None:
            return {}
        return self._normalise_stimulus(stimulus)

    def step(
        self,
        stimulus: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> StepResult:
        """Advance one bounded synchronous LIF-style step.

        Incoming edges read the previous step's spikes, so all nodes update
        from the same state and edge-list ordering cannot affect causality.
        Pending stimulation and the optional ``stimulus`` argument are applied
        to this step and then cleared/consumed.
        """

        immediate = self._step_stimulus(stimulus)
        external = {node_id: amount for node_id, amount in self._pending_stimulation.items()}
        for node_id, amount in immediate.items():
            external[node_id] = self._saturate(external[node_id] + amount)
        self._pending_stimulation = {node.id: 0.0 for node in self._nodes}

        previous_spikes = self._spikes
        previous_membrane = self._membrane
        next_membrane: dict[str, float] = {}
        next_spikes: dict[str, float] = {}
        for node in self._nodes:
            node_id = node.id
            if self._refractory[node_id] > 0:
                next_membrane[node_id] = node.reset
                next_spikes[node_id] = 0.0
                self._refractory[node_id] -= 1
                continue

            terms = [node.bias, external[node_id]]
            terms.extend(
                weight
                for source, weight in self._incoming[node_id]
                if previous_spikes[source] != 0.0
            )
            bounded_drive = self._saturate(math.fsum(self._saturate(term) for term in terms))
            candidate = node.reset + node.leak * (previous_membrane[node_id] - node.reset)
            candidate = self._saturate(candidate + bounded_drive)
            if candidate >= node.threshold:
                next_membrane[node_id] = node.reset
                next_spikes[node_id] = 1.0
                self._refractory[node_id] = node.refractory_steps
            else:
                next_membrane[node_id] = candidate
                next_spikes[node_id] = 0.0

        self._membrane = next_membrane
        self._spikes = next_spikes
        self._step_index += 1
        activity = _read_only(next_spikes)
        membrane = _read_only(next_membrane)
        return StepResult(
            step_index=self._step_index,
            activity=activity,
            membrane=membrane,
            spikes=activity,
        )

    def run(
        self,
        steps: int,
        *,
        stimulus: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        stimulation: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
    ) -> tuple[StepResult, ...]:
        """Run at most ``max_steps`` updates and return immutable snapshots.

        If supplied, ``stimulus`` is applied before every update.  The
        ``stimulation`` keyword is an explicit alias; passing both is an
        error.  A run of zero steps is valid and leaves state unchanged.
        """

        _integer(steps, "steps", minimum=0, maximum=self.max_steps)
        if stimulus is not None and stimulation is not None:
            raise ConnectomeValidationError("pass either stimulus or stimulation, not both")
        repeated = stimulus if stimulus is not None else stimulation
        if repeated is not None:
            # Validate once before mutating state, then reuse the canonical
            # mapping for every step.
            repeated = self._normalise_stimulus(repeated)
        return tuple(self.step(repeated) for _ in range(steps))

    def simulate(
        self,
        stimulus: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        *,
        steps: int = 1,
    ) -> tuple[StepResult, ...]:
        """Convenience alias for ``run(steps, stimulus=stimulus)``."""

        return self.run(steps, stimulus=stimulus)

    def readout(self, *, source: str = "spikes") -> Mapping[str, float]:
        """Return a read-only named readout for configured output nodes.

        ``source="spikes"`` (the default) returns the latest binary output
        activity.  ``source="membrane"`` returns the latest bounded membrane
        potentials.  Use :meth:`readout_vector` when a positional tuple is
        more convenient.
        """

        values = self._values_for(source)
        return _read_only({node_id: values[node_id] for node_id in self._outputs})

    def readout_vector(self, *, source: str = "spikes") -> tuple[float, ...]:
        """Return output values in the configured ``outputs`` order."""

        values = self._values_for(source)
        return tuple(values[node_id] for node_id in self._outputs)

    def _values_for(self, source: str) -> Mapping[str, float]:
        if source in {"spikes", "activity"}:
            return self._spikes
        if source in {"membrane", "potentials"}:
            return self._membrane
        raise ConnectomeValidationError("source must be 'spikes' or 'membrane'")


def load_connectome(
    source: JsonSource,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    potential_limit: float = DEFAULT_POTENTIAL_LIMIT,
) -> Connectome:
    """Load a validated :class:`Connectome` from the documented JSON format."""

    return Connectome.from_json(
        source,
        max_steps=max_steps,
        potential_limit=potential_limit,
    )


load_edge_list = load_connectome


__all__ = [
    "Connectome",
    "ConnectomeError",
    "ConnectomeFormatError",
    "ConnectomeValidationError",
    "Edge",
    "EdgeSpec",
    "Node",
    "NodeSpec",
    "StepResult",
    "load_connectome",
    "load_edge_list",
]
