"""Flychess: a fly-brain chess experiment."""

from .brain import DecisionReadout, MoveCandidate, SurrogateFlyBrain, encode_board
from .connectome import Connectome
from .connectome_policy import ConnectomeFlyBrain, ConnectomePolicyError
from .engine import StockfishEngine
from .experiment import ReplayedExperiment, read_experiment, write_experiment
from .flynet import FlyNetGraph, FlyNetModel, FlyNetPolicy, encode_flynet_board, generate_flynet_graph
from .flynet_training import FlyNetDataset, generate_teacher_dataset, load_flynet_model, train_flynet
from .game import DecisionRecord, GameResult, MoveRecord, ProgressCallback, play_game

__all__ = [
    "Connectome",
    "ConnectomeFlyBrain",
    "ConnectomePolicyError",
    "DecisionReadout",
    "DecisionRecord",
    "GameResult",
    "FlyNetDataset",
    "FlyNetGraph",
    "FlyNetModel",
    "FlyNetPolicy",
    "MoveCandidate",
    "MoveRecord",
    "ProgressCallback",
    "ReplayedExperiment",
    "StockfishEngine",
    "SurrogateFlyBrain",
    "encode_board",
    "encode_flynet_board",
    "generate_flynet_graph",
    "generate_teacher_dataset",
    "load_flynet_model",
    "play_game",
    "read_experiment",
    "write_experiment",
    "train_flynet",
]

__version__ = "0.1.0"
