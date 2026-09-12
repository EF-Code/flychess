"""Optional Feather-backed MaleCNS ingestion."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from .canonical import CanonicalConnectome, canonicalize_malecns_rows
from .manifest import DatasetManifest


class CalibrationDependencyError(RuntimeError):
    """Raised when optional bulk-data dependencies are unavailable."""


def iter_feather_rows(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    batch_size: int = 8_192,
) -> Iterator[Mapping[str, Any]]:
    """Stream rows from an Arrow Feather file in bounded record batches.

    ``pyarrow`` is intentionally optional: the core Flychess install remains
    lightweight, while a calibration environment can add ``pyarrow`` when it
    has acquired the large MaleCNS artifacts.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    try:
        import pyarrow as pa
        import pyarrow.ipc
    except ImportError as exc:
        raise CalibrationDependencyError(
            "MaleCNS Feather ingestion requires optional dependency 'pyarrow'"
        ) from exc

    source_path = Path(path)
    try:
        # Feather v2 is an Arrow IPC file. Opening it as a file reader avoids
        # materializing the whole table before the canonicalizer can filter it.
        with pa.memory_map(str(source_path), "r") as source:
            reader = pa.ipc.open_file(source)
            for index in range(reader.num_record_batches):
                batch = reader.get_batch(index)
                if columns is not None:
                    batch = batch.select(columns)
                for offset in range(0, batch.num_rows, batch_size):
                    yield from batch.slice(offset, batch_size).to_pylist()
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise CalibrationDependencyError(f"could not read Feather artifact {source_path}: {exc}") from exc


def import_malecns(
    manifest: DatasetManifest,
    *,
    root: str | Path,
    body_ids: Iterable[int] | None = None,
    batch_size: int = 8_192,
) -> CanonicalConnectome:
    """Import a filtered MaleCNS graph using manifest-declared artifacts.

    The manifest is verified before any rows are read. Expected artifact roles
    are ``annotations``, ``connectivity``, and optional
    ``neurotransmitters``. This importer is a data boundary; it does not fit
    neural dynamics or a chess readout.
    """

    manifest.verify(root)
    annotations = iter_feather_rows(
        manifest.artifact_path("annotations", root),
        batch_size=batch_size,
    )
    connections = iter_feather_rows(
        manifest.artifact_path("connectivity", root),
        batch_size=batch_size,
    )
    neurotransmitters: Iterable[Mapping[str, Any]] = ()
    try:
        neurotransmitter_path = manifest.artifact_path("neurotransmitters", root)
    except ValueError:
        pass
    else:
        neurotransmitters = iter_feather_rows(neurotransmitter_path, batch_size=batch_size)
    return canonicalize_malecns_rows(
        manifest=manifest,
        annotations=annotations,
        connections=connections,
        neurotransmitters=neurotransmitters,
        body_ids=body_ids,
    )
