"""Held-out MaleCNS parameter fitting for the bounded SparseLIF backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

from .calibration.canonical import CanonicalConnectome
from .calibration.dynamics import LIFParameters, SparseLIF
from .calibration.manifest import DatasetManifest


class MaleCNSCalibrationError(ValueError):
    """Raised when a source-locked calibration request is invalid."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MaleCNSCalibrationError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    input_current: Mapping[int, float]
    target_activity: Mapping[int, float]

    def __post_init__(self) -> None:
        for label, values in (("input_current", self.input_current), ("target_activity", self.target_activity)):
            if not isinstance(values, Mapping):
                raise MaleCNSCalibrationError(f"{label} must be a mapping")
            for body_id, value in values.items():
                if isinstance(body_id, bool) or not isinstance(body_id, int):
                    raise MaleCNSCalibrationError(f"{label} contains a non-integer body ID")
                _finite(value, f"{label}[{body_id}]")


@dataclass(frozen=True, slots=True)
class CalibrationTrace:
    steps: tuple[CalibrationSample, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise MaleCNSCalibrationError("calibration traces must contain at least one step")


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    manifest: DatasetManifest
    traces: tuple[CalibrationTrace, ...]

    def __post_init__(self) -> None:
        if not self.traces:
            raise MaleCNSCalibrationError("calibration dataset must contain at least one trace")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    train_fraction: float = 0.8
    leak_values: tuple[float, ...] = (0.2, 0.5, 0.8)
    weight_scale_values: tuple[float, ...] = (0.05, 0.1, 0.2)
    synapse_exponent_values: tuple[float, ...] = (0.5, 1.0)
    threshold: float = 1.0
    max_steps: int = 1000
    require_malecns_source: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise MaleCNSCalibrationError("train_fraction must be between zero and one")
        if not self.leak_values or not self.weight_scale_values or not self.synapse_exponent_values:
            raise MaleCNSCalibrationError("each calibration search range must be non-empty")
        if any(not 0.0 <= value < 1.0 for value in self.leak_values):
            raise MaleCNSCalibrationError("leak values must be in [0, 1)")
        if any(value < 0.0 for value in self.weight_scale_values):
            raise MaleCNSCalibrationError("weight scales must be non-negative")
        if any(value <= 0.0 for value in self.synapse_exponent_values):
            raise MaleCNSCalibrationError("synapse exponents must be positive")
        if self.threshold <= 0.0 or self.max_steps < 1:
            raise MaleCNSCalibrationError("threshold and max_steps must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    source: str
    dataset_id: str
    train_mae: float
    validation_mae: float
    parameters: Mapping[str, float]
    train_traces: int
    validation_traces: int
    candidates_evaluated: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dataset_id": self.dataset_id,
            "train_mae": self.train_mae,
            "validation_mae": self.validation_mae,
            "parameters": dict(self.parameters),
            "train_traces": self.train_traces,
            "validation_traces": self.validation_traces,
            "candidates_evaluated": self.candidates_evaluated,
        }


def _mae(
    graph: CanonicalConnectome,
    traces: Sequence[CalibrationTrace],
    *,
    inputs: Sequence[int],
    outputs: Sequence[int],
    leak: float,
    weight_scale: float,
    synapse_exponent: float,
    threshold: float,
    max_steps: int,
) -> float:
    parameters = {body_id: LIFParameters(threshold=threshold, leak=leak) for body_id in graph.node_ids}
    dynamics = SparseLIF(
        graph,
        inputs=inputs,
        outputs=outputs,
        node_parameters=parameters,
        weight_scale=weight_scale,
        synapse_exponent=synapse_exponent,
        max_steps=max_steps,
    )
    errors: list[float] = []
    for trace in traces:
        dynamics.reset()
        for sample in trace.steps:
            result = dynamics.step(sample.input_current)
            for body_id, expected in sample.target_activity.items():
                if body_id not in outputs:
                    raise MaleCNSCalibrationError(f"target body ID {body_id} is not an output port")
                errors.append(abs(float(result.activity.get(body_id, 0.0)) - float(expected)))
    return sum(errors) / len(errors) if errors else float("inf")


def fit_malecns_parameters(
    graph: CanonicalConnectome,
    dataset: CalibrationDataset,
    *,
    inputs: Sequence[int],
    outputs: Sequence[int],
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Fit bounded LIF parameters on train traces and report held-out MAE.

    This transparent grid search is suitable for a small calibration subgraph
    or precomputed summary, not for claiming that chess outcomes identify
    biological parameters.
    """
    config = config or CalibrationConfig()
    source = str(dataset.manifest.source)
    if config.require_malecns_source and source.lower() != "malecns":
        raise MaleCNSCalibrationError(
            f"activity calibration requires manifest source 'MaleCNS', got {source!r}; FlyWire is not interchangeable"
        )
    if len(dataset.traces) < 2:
        raise MaleCNSCalibrationError("at least two traces are required for a held-out split")
    if not inputs or not outputs:
        raise MaleCNSCalibrationError("inputs and outputs must be non-empty")
    train_count = min(len(dataset.traces) - 1, max(1, int(len(dataset.traces) * config.train_fraction)))
    train = dataset.traces[:train_count]
    validation = dataset.traces[train_count:]
    best: tuple[float, dict[str, float]] | None = None
    candidates = 0
    for leak, weight_scale, exponent in product(config.leak_values, config.weight_scale_values, config.synapse_exponent_values):
        candidates += 1
        params = {
            "leak": float(leak),
            "weight_scale": float(weight_scale),
            "synapse_exponent": float(exponent),
            "threshold": float(config.threshold),
        }
        score = _mae(graph, train, inputs=inputs, outputs=outputs, max_steps=config.max_steps, **params)
        if best is None or score < best[0] or (score == best[0] and tuple(params.values()) < tuple(best[1].values())):
            best = (score, params)
    assert best is not None
    validation_score = _mae(graph, validation, inputs=inputs, outputs=outputs, max_steps=config.max_steps, **best[1])
    return CalibrationResult(
        source=source,
        dataset_id=str(dataset.manifest.dataset_id),
        train_mae=best[0],
        validation_mae=validation_score,
        parameters=best[1],
        train_traces=len(train),
        validation_traces=len(validation),
        candidates_evaluated=candidates,
    )


__all__ = ["CalibrationConfig", "CalibrationDataset", "CalibrationResult", "CalibrationSample", "CalibrationTrace", "MaleCNSCalibrationError", "fit_malecns_parameters"]
