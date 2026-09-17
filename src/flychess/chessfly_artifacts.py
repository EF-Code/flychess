"""Provenance-locked acquisition and verification for ChessFly artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


class ChessFlyArtifactError(RuntimeError):
    """Raised when an artifact cannot be acquired or fails verification."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_bytes(path: Path) -> bytes:
    try:
        payload = path.read_bytes()
        return gzip.decompress(payload) if path.name.endswith(".gz") else payload
    except (OSError, EOFError) as exc:
        raise ChessFlyArtifactError(f"could not decode artifact {path}: {exc}") from exc


def _safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ChessFlyArtifactError(f"manifest path escapes artifact root: {relative!r}")
    return candidate


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    name: str
    relative_path: str
    url: str
    file_bytes: int
    file_sha256: str
    raw_bytes: int | None = None
    raw_sha256: str | None = None
    expected_raw_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ChessFlyArtifactManifest:
    manifest_version: int
    acquired_at_utc: str
    dataset: str
    space_revision: str
    model_revision: str
    files: tuple[ArtifactRecord, ...]
    source_meta_url: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["files"] = [asdict(record) for record in self.files]
        return value

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def read(cls, path: str | Path) -> "ChessFlyArtifactManifest":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            files = tuple(ArtifactRecord(**record) for record in payload["files"])
            return cls(
                manifest_version=int(payload["manifest_version"]),
                acquired_at_utc=str(payload["acquired_at_utc"]),
                dataset=str(payload["dataset"]),
                space_revision=str(payload["space_revision"]),
                model_revision=str(payload["model_revision"]),
                files=files,
                source_meta_url=str(payload["source_meta_url"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ChessFlyArtifactError(f"invalid ChessFly manifest {path}: {exc}") from exc

    def verify(self, root: str | Path) -> None:
        artifact_root = Path(root).resolve()
        for record in self.files:
            path = _safe_path(artifact_root, record.relative_path)
            if not path.is_file():
                raise ChessFlyArtifactError(f"missing artifact {path}")
            if path.stat().st_size != record.file_bytes:
                raise ChessFlyArtifactError(f"size mismatch for {record.name}")
            if _sha256_path(path) != record.file_sha256:
                raise ChessFlyArtifactError(f"SHA-256 mismatch for {record.name}")
            if record.raw_sha256 is not None:
                raw = _raw_bytes(path)
                if record.raw_bytes is not None and len(raw) != record.raw_bytes:
                    raise ChessFlyArtifactError(f"decoded size mismatch for {record.name}")
                actual_raw_sha256 = hashlib.sha256(raw).hexdigest()
                if actual_raw_sha256 != record.raw_sha256:
                    raise ChessFlyArtifactError(f"recorded decoded SHA-256 mismatch for {record.name}")
                if record.expected_raw_sha256 is not None and actual_raw_sha256 != record.expected_raw_sha256:
                    raise ChessFlyArtifactError(f"published decoded SHA-256 mismatch for {record.name}")


def _download_atomic(url: str, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as output:
            request = Request(url, headers={"User-Agent": "flychess-artifact-loader/0.1"})
            try:
                with urlopen(request) as response:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            except Exception as exc:
                raise ChessFlyArtifactError(f"download failed for {url}: {exc}") from exc
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _record(root: Path, name: str, relative_path: str, url: str, expected_raw_sha256: str | None) -> ArtifactRecord:
    path = _safe_path(root, relative_path)
    raw = _raw_bytes(path) if path.name.endswith(".gz") else None
    return ArtifactRecord(
        name=name,
        relative_path=relative_path,
        url=url,
        file_bytes=path.stat().st_size,
        file_sha256=_sha256_path(path),
        raw_bytes=None if raw is None else len(raw),
        raw_sha256=None if raw is None else hashlib.sha256(raw).hexdigest(),
        expected_raw_sha256=expected_raw_sha256,
    )


def acquire_chessfly_artifacts(
    root: str | Path,
    *,
    space_revision: str = "main",
    model_revision: str = "main",
    force: bool = False,
) -> ChessFlyArtifactManifest:
    """Download public graph/weights and write a reproducible manifest.

    Graph files are checked against the Space published decoded hashes. The
    weight hash is recorded after acquisition because the model card does not
    expose an independent whole-file digest. Use immutable Hub commit IDs for
    both revisions when creating a release manifest; main is experiment-only.
    """
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    meta_url = f"https://huggingface.co/spaces/mlabonne/chessfly/resolve/{space_revision}/data/meta.json"
    try:
        meta = json.loads(urlopen(meta_url).read())
    except Exception as exc:
        raise ChessFlyArtifactError(f"could not read Space metadata {meta_url}: {exc}") from exc
    published_files = meta.get("files", {})
    graph_specs = (
        ("connectome.bin.gz", published_files.get("connectome.bin.gz", {}).get("rawSha256")),
        ("neurons.bin.gz", published_files.get("neurons.bin.gz", {}).get("rawSha256")),
    )
    records: list[ArtifactRecord] = []
    for name, expected_hash in graph_specs:
        url = f"https://huggingface.co/spaces/mlabonne/chessfly/resolve/{space_revision}/data/{name}"
        _download_atomic(url, root_path / name, force=force)
        records.append(_record(root_path, name, name, url, expected_hash))
    weights_url = f"https://huggingface.co/mlabonne/chessfly/resolve/{model_revision}/flynet.safetensors?download=true"
    _download_atomic(weights_url, root_path / "flynet.safetensors", force=force)
    records.append(_record(root_path, "flynet.safetensors", "flynet.safetensors", weights_url, None))
    manifest = ChessFlyArtifactManifest(
        manifest_version=1,
        acquired_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset=str(meta.get("dataset", "unknown")),
        space_revision=space_revision,
        model_revision=model_revision,
        files=tuple(records),
        source_meta_url=meta_url,
    )
    manifest.verify(root_path)
    manifest.write(root_path / "manifest.json")
    return manifest


__all__ = ["ArtifactRecord", "ChessFlyArtifactError", "ChessFlyArtifactManifest", "acquire_chessfly_artifacts"]
