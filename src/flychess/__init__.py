"""Flychess: a fly-brain chess experiment."""

from .brain import SurrogateFlyBrain, encode_board
from .connectome import Connectome
from .connectome_policy import ConnectomeFlyBrain, ConnectomePolicyError
from .experiment import ReplayedExperiment, read_experiment, write_experiment

__all__ = [
    "Connectome",
    "ConnectomeFlyBrain",
    "ConnectomePolicyError",
    "ReplayedExperiment",
    "SurrogateFlyBrain",
    "encode_board",
    "read_experiment",
    "write_experiment",
]

__version__ = "0.1.0"
