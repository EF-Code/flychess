"""Sparse, deterministic dynamics for imported connectome subgraphs.

This module supplies the first runtime layer after data ingestion. It keeps
all biological assumptions explicit: source weights are transformed by a
configurable scale and exponent, missing synapse signs use an explicit
default, and every neuron starts with the same parameters unless a caller
provides cell-specific overrides. It is a calibration target, not a claim
that these parameters reproduce a living fly.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .canonical import CanonicalConnectome


DEFAULT_MAX_STEPS = 1_000
DEFAULT_POTENTIAL_LIMIT = 10.0
MAX_ALLOWED_STEPS = 1_000_000


class DynamicsError(ValueError):
    """Raised when a sparse dynamics configuration or stimulus is invalid."""


class DynamicsDependencyError(DynamicsError):
    """Reserved for future optional numerical backends."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DynamicsError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise DynamicsError(f"{label} must be a finite number")
    return converted


def _integer(value: Any, label: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DynamicsError(f"{label} must be an integer")
    if value < minimum or maximum is not None and value > maximum:
        bound = f"at most {maximum}" if maximum is not None and value > maximum else f"at least {minimum}"
        raise DynamicsError(f"{label} must be {bound}")
    return value


@dataclass(frozen=True, slots=True)
class LIFParameters:
    """Parameters for one leaky-integrate-and-fire neuron."""

    threshold: float = 1.0
    leak: float = 0.8
    reset: float = 0.0
    bias: float = 0.0
    refractory_steps: int = 0

    def __post_init__(self) -> None:
        threshold = _finite(self.threshold, "threshold")
        leak = _finite(self.leak, "leak")
        reset = _finite(self.reset, "reset")
        bias = _finite(self.bias, "bias")
        refractory = _integer(self.refractory_steps, "refractory_steps", minimum=0)
        if threshold <= 0.0:
            raise DynamicsError("threshold must be greater than zero")
        if not 0.0 <= leak < 1.0:
            raise DynamicsError("leak must be in [0, 1)")
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "leak", leak)
        object.__setattr__(self, "reset", reset)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "refractory_steps", refractory)


@dataclass(frozen=True, slots=True)
class SparseStepResult:
    """Immutable snapshots emitted by :class:`SparseLIF`."""

    step_index: int
    activity: Mapping[int, float]
    membrane: Mapping[int, float]
    spikes: Mapping[int, float]

    def __getitem__(self, body_id: int) -> float:
        return self.activity[body_id]


def _read_only(values: Mapping[int, float]) -> Mapping[int, float]:
    return MappingProxyType(dict(values))


class SparseLIF:
    """Run bounded synchronous LIF updates over a canonical subgraph.

    ``synapse_exponent`` and ``weight_scale`` turn source weights into
    simulation currents as ``weight_scale * abs(source_weight) **
    synapse_exponent``. An explicit edge sign takes precedence; otherwise
    ``default_sign`` is used (and the sign of a negative source weight is
    preserved). This transform is intentionally exposed for later fitting.
    """

    def __init__(
        self,
        graph: CanonicalConnectome,
        *,
        inputs: Iterable[int] | None = None,
        outputs: Iterable[int] | None = None,
        node_parameters: Mapping[int, LIFParameters] | None = None,
        weight_scale: float = 0.1,
        synapse_exponent: float = 0.5,
        default_sign: float = 1.0,
        max_steps: int = DEFAULT_MAX_STEPS,
        potential_limit: float = DEFAULT_POTENTIAL_LIMIT,
    ) -> None:
        if not isinstance(graph, CanonicalConnectome):
            raise DynamicsError("graph must be a CanonicalConnectome")
        self._node_ids = graph.node_ids
        node_id_set = set(self._node_ids)
        self._inputs = self._ports(inputs, "inputs", node_id_set, default=self._node_ids)
        self._outputs = self._ports(outputs, "outputs", node_id_set, default=self._node_ids)

        self.weight_scale = _finite(weight_scale, "weight_scale")
        if self.weight_scale < 0.0:
            raise DynamicsError("weight_scale must not be negative")
        self.synapse_exponent = _finite(synapse_exponent, "synapse_exponent")
        if self.synapse_exponent <= 0.0:
            raise DynamicsError("synapse_exponent must be greater than zero")
        self.default_sign = _finite(default_sign, "default_sign")
        if not -1.0 <= self.default_sign <= 1.0:
            raise DynamicsError("default_sign must be in [-1, 1]")
        self.potential_limit = _finite(potential_limit, "potential_limit")
        if self.potential_limit <= 0.0:
            raise DynamicsError("potential_limit must be greater than zero")
        self.max_steps = _integer(
            max_steps,
            "max_steps",
            minimum=1,
            maximum=MAX_ALLOWED_STEPS,
        )

        parameters = {body_id: LIFParameters() for body_id in self._node_ids}
        if node_parameters is not None:
            for body_id, value in node_parameters.items():
                if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id not in node_id_set:
                    raise DynamicsError(f"node_parameters references unknown body ID {body_id}")
                if not isinstance(value, LIFParameters):
                    raise DynamicsError("node_parameters values must be LIFParameters")
                parameters[body_id] = value
        if any(abs(value.reset) > self.potential_limit for value in parameters.values()):
            raise DynamicsError("node reset must be within potential_limit")
        if any(value.threshold > self.potential_limit for value in parameters.values()):
            raise DynamicsError("node threshold must not exceed potential_limit")
        self._parameters = parameters

        incoming: dict[int, list[tuple[int, float]]] = {body_id: [] for body_id in self._node_ids}
        for edge in graph.connections:
            raw_weight = edge.effective_weight
            magnitude = abs(raw_weight) ** self.synapse_exponent
            sign = edge.sign if edge.sign is not None else self.default_sign
            if edge.sign is None and raw_weight < 0.0:
                sign = -sign
            incoming[edge.post_body_id].append(
                (edge.pre_body_id, self._saturate(self.weight_scale * magnitude * sign))
            )
        self._incoming = {
            body_id: tuple(sorted(edges, key=lambda edge: (edge[0], edge[1])))
            for body_id, edges in incoming.items()
        }
        self._effective_edges = tuple(
            (source, target, weight)
            for target in self._node_ids
            for source, weight in self._incoming[target]
        )
        self._step_index = 0
        self._membrane: dict[int, float] = {}
        self._spikes: dict[int, float] = {}
        self._refractory: dict[int, int] = {}
        self._pending_stimulation: dict[int, float] = {}
        self.reset()

    @staticmethod
    def _ports(
        values: Iterable[int] | None,
        label: str,
        node_ids: set[int],
        *,
        default: Sequence[int],
    ) -> tuple[int, ...]:
        if values is None:
            return tuple(default)
        if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Iterable):
            raise DynamicsError(f"{label} must be an iterable of body IDs")
        ports: list[int] = []
        seen: set[int] = set()
        for body_id in values:
            if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id not in node_ids:
                raise DynamicsError(f"{label} references unknown body ID {body_id}")
            if body_id in seen:
                raise DynamicsError(f"{label} contains duplicate body ID {body_id}")
            seen.add(body_id)
            ports.append(body_id)
        return tuple(ports)

    @property
    def node_ids(self) -> tuple[int, ...]:
        return self._node_ids

    @property
    def inputs(self) -> tuple[int, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[int, ...]:
        return self._outputs

    @property
    def effective_edges(self) -> tuple[tuple[int, int, float], ...]:
        """Return canonical ``(pre, post, current_weight)`` triples."""

        return self._effective_edges

    @property
    def steps_run(self) -> int:
        return self._step_index

    @property
    def node_activity(self) -> Mapping[int, float]:
        return _read_only(self._spikes)

    @property
    def membrane_potentials(self) -> Mapping[int, float]:
        return _read_only(self._membrane)

    def reset(self) -> None:
        self._step_index = 0
        self._membrane = {body_id: self._parameters[body_id].reset for body_id in self._node_ids}
        self._spikes = {body_id: 0.0 for body_id in self._node_ids}
        self._refractory = {body_id: 0 for body_id in self._node_ids}
        self._pending_stimulation = {body_id: 0.0 for body_id in self._node_ids}

    def _saturate(self, value: float) -> float:
        if not math.isfinite(value):
            return math.copysign(self.potential_limit, value)
        return min(self.potential_limit, max(-self.potential_limit, value))

    def _normalise_stimulus(
        self,
        stimulus: Mapping[int, Any] | Iterable[tuple[int, Any]],
        *,
        label: str = "stimulus",
    ) -> dict[int, float]:
        if isinstance(stimulus, Mapping):
            items = iter(stimulus.items())
        else:
            if isinstance(stimulus, (str, bytes, bytearray)) or not isinstance(stimulus, Iterable):
                raise DynamicsError(f"{label} must map input body IDs to numeric currents")
            items = iter(stimulus)
        values: dict[int, float] = {}
        input_ids = set(self._inputs)
        for index, item in enumerate(items):
            if (
                isinstance(item, (str, bytes, bytearray))
                or not isinstance(item, Sequence)
                or len(item) != 2
            ):
                raise DynamicsError(f"{label}[{index}] must be a (body_id, value) pair")
            body_id, raw_value = item
            if isinstance(body_id, bool) or not isinstance(body_id, int) or body_id not in input_ids:
                raise DynamicsError(f"{label} references non-input body ID {body_id}")
            amount = _finite(raw_value, f"{label}[{index}] value")
            values[body_id] = self._saturate(values.get(body_id, 0.0) + amount)
        return values

    def stimulate(
        self,
        stimulus: Mapping[int, Any] | Iterable[tuple[int, Any]] | int,
        value: Any | None = None,
    ) -> None:
        """Queue a one-step stimulus for one or more configured inputs."""

        if isinstance(stimulus, int) and not isinstance(stimulus, bool):
            if value is None:
                raise DynamicsError("stimulate(body_id, value) requires a value")
            values = self._normalise_stimulus(((stimulus, value),))
        else:
            if value is not None:
                raise DynamicsError("value is only valid with a single body ID")
            values = self._normalise_stimulus(stimulus)
        for body_id, amount in values.items():
            self._pending_stimulation[body_id] = self._saturate(
                self._pending_stimulation[body_id] + amount
            )

    def step(
        self,
        stimulus: Mapping[int, Any] | Iterable[tuple[int, Any]] | None = None,
    ) -> SparseStepResult:
        """Advance one synchronous bounded update using previous-step spikes."""

        immediate = {} if stimulus is None else self._normalise_stimulus(stimulus)
        external = dict(self._pending_stimulation)
        for body_id, amount in immediate.items():
            external[body_id] = self._saturate(external[body_id] + amount)
        self._pending_stimulation = {body_id: 0.0 for body_id in self._node_ids}

        previous_spikes = self._spikes
        previous_membrane = self._membrane
        next_membrane: dict[int, float] = {}
        next_spikes: dict[int, float] = {}
        for body_id in self._node_ids:
            parameters = self._parameters[body_id]
            if self._refractory[body_id] > 0:
                next_membrane[body_id] = parameters.reset
                next_spikes[body_id] = 0.0
                self._refractory[body_id] -= 1
                continue

            incoming = sum(
                weight
                for source, weight in self._incoming[body_id]
                if previous_spikes[source] != 0.0
            )
            drive = self._saturate(parameters.bias + external[body_id] + incoming)
            candidate = parameters.reset + parameters.leak * (
                previous_membrane[body_id] - parameters.reset
            )
            candidate = self._saturate(candidate + drive)
            if candidate >= parameters.threshold:
                next_membrane[body_id] = parameters.reset
                next_spikes[body_id] = 1.0
                self._refractory[body_id] = parameters.refractory_steps
            else:
                next_membrane[body_id] = candidate
                next_spikes[body_id] = 0.0

        self._membrane = next_membrane
        self._spikes = next_spikes
        self._step_index += 1
        activity = _read_only(next_spikes)
        return SparseStepResult(
            step_index=self._step_index,
            activity=activity,
            membrane=_read_only(next_membrane),
            spikes=activity,
        )

    def run(
        self,
        steps: int,
        *,
        stimulus: Mapping[int, Any] | Iterable[tuple[int, Any]] | None = None,
    ) -> tuple[SparseStepResult, ...]:
        """Run a caller-bounded number of updates with repeated stimulus."""

        _integer(steps, "steps", minimum=0, maximum=self.max_steps)
        repeated = None if stimulus is None else self._normalise_stimulus(stimulus)
        return tuple(self.step(repeated) for _ in range(steps))

    def readout(self, *, source: str = "spikes") -> Mapping[int, float]:
        values = self._values_for(source)
        return _read_only({body_id: values[body_id] for body_id in self._outputs})

    def readout_vector(self, *, source: str = "spikes") -> tuple[float, ...]:
        values = self._values_for(source)
        return tuple(values[body_id] for body_id in self._outputs)

    def _values_for(self, source: str) -> Mapping[int, float]:
        if source in {"spikes", "activity"}:
            return self._spikes
        if source in {"membrane", "potentials"}:
            return self._membrane
        raise DynamicsError("source must be 'spikes' or 'membrane'")


__all__ = [
    "DynamicsDependencyError",
    "DynamicsError",
    "LIFParameters",
    "SparseLIF",
    "SparseStepResult",
]
