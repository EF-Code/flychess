"""Dataset identity, artifact hashing, and local verification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a calibration manifest is malformed."""


class ManifestVerificationError(ManifestError):
    """Raised when a manifest cannot be verified against local artifacts."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ManifestError(f"{field} contains a NUL character")
    return value.strip()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local file without loading it into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ManifestVerificationError(f"could not read artifact {path}: {exc}") from exc
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Artifact:
    """One externally sourced calibration artifact."""

    role: str
    path: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        role = _text(self.role, "artifact role")
        path = _text(self.path, "artifact path")
        parsed = Path(path)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise ManifestError("artifact path must be relative and must not contain '..'")
        if self.sha256 is not None:
            digest = _text(self.sha256, "artifact sha256").lower()
            if not _SHA256.fullmatch(digest):
                raise ManifestError("artifact sha256 must be a 64-character lowercase hex digest")
            object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "path", path)

    def as_dict(self) -> dict[str, str]:
        payload = {"role": self.role, "path": self.path}
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        return payload


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Portable identity and artifact inventory for one connectome release."""

    source: str
    version: str
    sex: str
    dataset_id: str
    license: str
    artifacts: tuple[Artifact, ...]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field in ("source", "version", "sex", "dataset_id", "license"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        artifacts = tuple(self.artifacts)
        if not artifacts:
            raise ManifestError("manifest must contain at least one artifact")
        if not all(isinstance(artifact, Artifact) for artifact in artifacts):
            raise ManifestError("manifest artifacts must be Artifact values")
        roles = [artifact.role for artifact in artifacts]
        if len(set(roles)) != len(roles):
            raise ManifestError("manifest artifact roles must be unique")
        object.__setattr__(self, "artifacts", artifacts)

        metadata = tuple(self.metadata)
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in metadata):
            raise ManifestError("manifest metadata must contain key/value pairs")
        metadata = tuple(
            sorted(
                (_text(pair[0], "metadata key"), _text(pair[1], f"metadata {pair[0]}"))
                for pair in metadata
            )
        )
        if len({key for key, _value in metadata}) != len(metadata):
            raise ManifestError("manifest metadata keys must be unique")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        """Parse the strict JSON-compatible manifest shape."""

        if not isinstance(value, Mapping):
            raise ManifestError("manifest must be a JSON object")
        allowed = {"source", "version", "sex", "dataset_id", "license", "artifacts", "metadata"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ManifestError(f"manifest has unknown fields: {', '.join(map(str, unknown))}")
        raw_artifacts = value.get("artifacts")
        if isinstance(raw_artifacts, (str, bytes, bytearray)) or not isinstance(raw_artifacts, list):
            raise ManifestError("manifest artifacts must be a list")
        artifacts: list[Artifact] = []
        for index, raw in enumerate(raw_artifacts):
            if not isinstance(raw, Mapping):
                raise ManifestError(f"artifacts[{index}] must be an object")
            artifact_allowed = {"role", "path", "sha256"}
            unknown_artifact = sorted(set(raw) - artifact_allowed)
            if unknown_artifact:
                raise ManifestError(
                    f"artifacts[{index}] has unknown fields: {', '.join(map(str, unknown_artifact))}"
                )
            artifacts.append(
                Artifact(
                    role=raw.get("role"),
                    path=raw.get("path"),
                    sha256=raw.get("sha256"),
                )
            )
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ManifestError("manifest metadata must be an object")
        metadata = tuple(
            sorted(
                (_text(key, "metadata key"), _text(item, f"metadata {key}"))
                for key, item in raw_metadata.items()
            )
        )
        return cls(
            source=value.get("source"),
            version=value.get("version"),
            sex=value.get("sex"),
            dataset_id=value.get("dataset_id"),
            license=value.get("license"),
            artifacts=tuple(artifacts),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, source: str | Path) -> "DatasetManifest":
        path = Path(source)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"could not read manifest {path}: {exc}") from exc
        return cls.from_mapping(value)

    @property
    def is_locked(self) -> bool:
        """Whether every artifact has a content hash."""

        return all(artifact.sha256 is not None for artifact in self.artifacts)

    def artifact(self, role: str) -> Artifact:
        role = _text(role, "artifact role")
        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        raise ManifestError(f"manifest has no artifact with role {role!r}")

    def artifact_path(self, role: str, root: str | Path) -> Path:
        """Resolve an artifact below *root* without allowing path escape."""

        root_path = Path(root).resolve()
        candidate = (root_path / self.artifact(role).path).resolve()
        if not candidate.is_relative_to(root_path):
            raise ManifestError(f"artifact {role!r} escapes manifest root")
        return candidate

    def verify(self, root: str | Path) -> None:
        """Verify existence and SHA-256 of every manifest artifact."""

        if not self.is_locked:
            raise ManifestVerificationError("manifest is not locked: every artifact needs sha256")
        for artifact in self.artifacts:
            path = self.artifact_path(artifact.role, root)
            if not path.is_file():
                raise ManifestVerificationError(f"artifact {artifact.role!r} is missing: {path}")
            actual = sha256_file(path)
            if actual != artifact.sha256:
                raise ManifestVerificationError(
                    f"artifact {artifact.role!r} hash mismatch: expected {artifact.sha256}, got {actual}"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "sex": self.sex,
            "dataset_id": self.dataset_id,
            "license": self.license,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "metadata": dict(self.metadata),
        }
