"""Provenance and ingestion primitives for biological Flychess backends.

This package deliberately stops before claiming biological calibration. It
locks dataset identity, normalizes a filtered connectome subgraph, and exposes
an explicit sparse dynamics layer while leaving fitting and task readouts for
later stages.
"""

from .canonical import (
    CanonicalConnectome,
    CanonicalizationError,
    ConnectionRecord,
    NeuronRecord,
    canonicalize_malecns_rows,
)
from .dynamics import (
    DynamicsDependencyError,
    DynamicsError,
    LIFParameters,
    SparseLIF,
    SparseStepResult,
)
from .malecns import CalibrationDependencyError, import_malecns, iter_feather_rows
from .manifest import (
    Artifact,
    DatasetManifest,
    ManifestError,
    ManifestVerificationError,
    sha256_file,
)

__all__ = [
    "Artifact",
    "CalibrationDependencyError",
    "CanonicalConnectome",
    "CanonicalizationError",
    "ConnectionRecord",
    "DynamicsDependencyError",
    "DynamicsError",
    "DatasetManifest",
    "LIFParameters",
    "ManifestError",
    "ManifestVerificationError",
    "NeuronRecord",
    "canonicalize_malecns_rows",
    "import_malecns",
    "iter_feather_rows",
    "sha256_file",
    "SparseLIF",
    "SparseStepResult",
]
