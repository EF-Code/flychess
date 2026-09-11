"""Flychess: a fly-brain chess experiment."""

from .brain import DecisionReadout, MoveCandidate, SurrogateFlyBrain, encode_board
from .connectome import Connectome
from .connectome_policy import ConnectomeFlyBrain, ConnectomePolicyError
from .engine import StockfishEngine
from .experiment import ReplayedExperiment, read_experiment, write_experiment
from .game import DecisionRecord, GameResult, MoveRecord, play_game

__all__ = [
    "Connectome",
    "ConnectomeFlyBrain",
    "ConnectomePolicyError",
    "DecisionReadout",
    "DecisionRecord",
    "GameResult",
    "MoveCandidate",
    "MoveRecord",
    "ReplayedExperiment",
    "StockfishEngine",
    "SurrogateFlyBrain",
    "encode_board",
    "play_game",
    "read_experiment",
    "write_experiment",
]

__version__ = "0.1.0"
