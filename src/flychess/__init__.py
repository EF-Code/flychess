"""Flychess: a fly-brain chess experiment."""

from .brain import SurrogateFlyBrain, encode_board
from .connectome import Connectome
from .connectome_policy import ConnectomeFlyBrain, ConnectomePolicyError

__all__ = [
    "Connectome",
    "ConnectomeFlyBrain",
    "ConnectomePolicyError",
    "SurrogateFlyBrain",
    "encode_board",
]

__version__ = "0.1.0"
