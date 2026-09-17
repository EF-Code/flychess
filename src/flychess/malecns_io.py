"""Validated Flychess interchange format for MaleCNS calibration targets.

This module defines a small interchange contract for exported MaleCNS activity
traces. It does not claim to parse an official provider export schema.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .calibration import DatasetManifest
from .malecns_calibration import CalibrationDataset, CalibrationSample, CalibrationTrace, MaleCNSCalibrationError


class MaleCNSCalibrationIOError(MaleCNSCalibrationError):
    """Raised when an interchange file or its provenance is invalid."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaleCNSCalibrationIOError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise MaleCNSCalibrationIOError(f"{label} must be a finite number")
    return result


def _parse_numeric_map(value: Any, label: str) -> dict[int, float]:
    if not isinstance(value, Mapping) or not value:
        raise MaleCNSCalibrationIOError(f"{label} must be a non-empty object")
    parsed: dict[int, float] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_key, bool):
            raise MaleCNSCalibrationIOError(f"{label} has a non-integer neuron id")
        if isinstance(raw_key, int):
            neuron_id = raw_key
        elif isinstance(raw_key, str):
            key = raw_key.strip()
            signless = key[1:] if key[:1] in {"+", "-"} else key
            if not signless or not signless.isdigit():
                raise MaleCNSCalibrationIOError(f"{label} has a non-integer neuron id: {raw_key!r}")
            neuron_id = int(key)
        else:
            raise MaleCNSCalibrationIOError(f"{label} has a non-integer neuron id: {raw_key!r}")
        if neuron_id in parsed:
            raise MaleCNSCalibrationIOError(f"{label} contains duplicate neuron id {neuron_id}")
        parsed[neuron_id] = _finite(raw_value, f"{label}[{neuron_id}]")
    return parsed


def _load_records(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise MaleCNSCalibrationIOError(f"trace file does not exist: {path}")
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            records: list[Mapping[str, Any]] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MaleCNSCalibrationIOError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
                if not isinstance(record, Mapping):
                    raise MaleCNSCalibrationIOError(f"line {line_number} must contain a JSON object")
                records.append(record)
            return records
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MaleCNSCalibrationIOError(f"unable to read trace file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise MaleCNSCalibrationIOError(f"invalid JSON in trace file: {exc.msg}") from exc
    if not isinstance(payload, list) or not all(isinstance(record, Mapping) for record in payload):
        raise MaleCNSCalibrationIOError("JSON calibration traces must be an array of objects")
    return payload


def _load_locked_malecns_manifest(manifest_path: Path, root: Path | None) -> DatasetManifest:
    try:
        manifest = DatasetManifest.from_json(manifest_path)
    except Exception as exc:
        raise MaleCNSCalibrationIOError(f"unable to load dataset manifest: {manifest_path}") from exc
    if manifest.source != "MaleCNS" or manifest.sex != "male":
        raise MaleCNSCalibrationIOError("calibration targets require a locked male MaleCNS manifest")
    if not manifest.is_locked:
        raise MaleCNSCalibrationIOError("calibration targets require sha256-locked manifest artifacts")
    if root is not None:
        try:
            manifest.verify(root)
        except Exception as exc:
            raise MaleCNSCalibrationIOError(f"dataset manifest verification failed under {root}") from exc
    return manifest


def load_malecns_calibration_dataset(trace_path: str | Path, manifest_path: str | Path, *, root: str | Path | None = None, max_traces: int = 100_000) -> CalibrationDataset:
    """Load validated MaleCNS targets into the calibration fitter.

    A trace has a non-empty samples array. Each sample has non-empty stimulus
    and target_activity objects mapping integer neuron IDs to finite numbers.
    Optional fields are trace_id, metadata, and non-negative integer step.
    Unknown fields are rejected.
    """
    if isinstance(max_traces, bool) or not isinstance(max_traces, int) or max_traces < 1:
        raise MaleCNSCalibrationIOError("max_traces must be a positive integer")
    dataset_root = Path(root) if root is not None else None
    manifest = _load_locked_malecns_manifest(Path(manifest_path), dataset_root)
    records = _load_records(Path(trace_path))
    if not records:
        raise MaleCNSCalibrationIOError("calibration trace file is empty")
    if len(records) > max_traces:
        raise MaleCNSCalibrationIOError(f"trace count {len(records)} exceeds max_traces={max_traces}")
    allowed_trace_keys = {"samples", "trace_id", "metadata"}
    allowed_sample_keys = {"stimulus", "target_activity", "step"}
    traces: list[CalibrationTrace] = []
    for trace_index, record in enumerate(records):
        unknown = set(record) - allowed_trace_keys
        if unknown:
            raise MaleCNSCalibrationIOError(f"trace {trace_index} has unknown fields: {sorted(unknown)!r}")
        samples = record.get("samples")
        if not isinstance(samples, list) or not samples:
            raise MaleCNSCalibrationIOError(f"trace {trace_index} must contain a non-empty samples array")
        parsed_samples: list[CalibrationSample] = []
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise MaleCNSCalibrationIOError(f"trace {trace_index} sample {sample_index} must be an object")
            unknown_sample = set(sample) - allowed_sample_keys
            if unknown_sample:
                raise MaleCNSCalibrationIOError(f"trace {trace_index} sample {sample_index} has unknown fields: {sorted(unknown_sample)!r}")
            if "stimulus" not in sample or "target_activity" not in sample:
                raise MaleCNSCalibrationIOError(f"trace {trace_index} sample {sample_index} needs stimulus and target_activity")
            step = sample.get("step")
            if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
                raise MaleCNSCalibrationIOError(f"trace {trace_index} sample {sample_index} step must be a non-negative integer")
            stimulus = _parse_numeric_map(sample["stimulus"], f"trace {trace_index} sample {sample_index} stimulus")
            target = _parse_numeric_map(sample["target_activity"], f"trace {trace_index} sample {sample_index} target_activity")
            parsed_samples.append(CalibrationSample(stimulus, target))
        traces.append(CalibrationTrace(tuple(parsed_samples)))
    return CalibrationDataset(manifest, tuple(traces))


__all__ = ["MaleCNSCalibrationIOError", "load_malecns_calibration_dataset"]
