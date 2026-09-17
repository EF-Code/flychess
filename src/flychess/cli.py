"""Command-line entry point for the first Flychess experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess

from .brain import SurrogateFlyBrain
from .connectome import load_connectome
from .connectome_policy import ConnectomeFlyBrain
from .engine import StockfishEngine
from .experiment import write_experiment
from .game import play_game


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flychess",
        description="Play a bounded chess game between a Flychess policy and Stockfish.",
    )
    parser.add_argument("--engine", default="stockfish", help="Stockfish executable path")
    parser.add_argument("--depth", type=int, default=4, help="Stockfish search depth")
    parser.add_argument("--fly-color", choices=("white", "black"), default="white")
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17, help="Deterministic surrogate-brain seed")
    parser.add_argument(
        "--connectome",
        type=Path,
        default=None,
        help="Use a validated connectome JSON edge list instead of the surrogate policy",
    )
    parser.add_argument(
        "--chessfly-connectome",
        type=Path,
        default=None,
        help="Public ChessFly connectome.bin.gz artifact",
    )
    parser.add_argument(
        "--chessfly-neurons",
        type=Path,
        default=None,
        help="Public ChessFly neurons.bin.gz artifact",
    )
    parser.add_argument(
        "--chessfly-weights",
        type=Path,
        default=None,
        help="Public ChessFly flynet.safetensors artifact",
    )
    parser.add_argument(
        "--chessfly-device",
        default="cpu",
        help="PyTorch device for the optional ChessFly policy (default: cpu)",
    )
    parser.add_argument(
        "--flynet-weights",
        type=Path,
        default=None,
        help="Flychess-trained FlyNet SafeTensors weights",
    )
    parser.add_argument(
        "--flynet-graph",
        type=Path,
        default=None,
        help="Flychess-trained FlyNet sparse graph (.npz)",
    )
    parser.add_argument(
        "--flynet-graph-metadata",
        type=Path,
        default=None,
        help="Optional FlyNet graph metadata JSON",
    )
    parser.add_argument(
        "--flynet-config",
        type=Path,
        default=None,
        help="Optional FlyNet model config JSON",
    )
    parser.add_argument(
        "--flynet-device",
        default="cpu",
        help="PyTorch device for the FlyNet policy (default: cpu)",
    )
    parser.add_argument(
        "--neural-steps",
        type=int,
        default=2,
        help="Connectome steps per fly turn (used with --connectome)",
    )
    parser.add_argument("--fen", default=None, help="Start from a FEN instead of the initial position")
    parser.add_argument("--record", type=Path, default=None, help="Write a replayable JSONL experiment log")
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="Optionally write the final result JSON (requires --record)",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only the final result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.result is not None and args.record is None:
        raise SystemExit("--result requires --record")
    board = chess.Board(args.fen) if args.fen else chess.Board()
    fly_color = chess.WHITE if args.fly_color == "white" else chess.BLACK
    chessfly_paths = (args.chessfly_connectome, args.chessfly_neurons, args.chessfly_weights)
    flynet_paths = (args.flynet_weights, args.flynet_graph)
    configured_policies = sum(
        (
            args.connectome is not None,
            any(path is not None for path in chessfly_paths),
            any(path is not None for path in flynet_paths),
        )
    )
    if configured_policies > 1:
        raise SystemExit("--connectome, --chessfly-*, and --flynet-* policies are mutually exclusive")
    if any(path is not None for path in chessfly_paths) and not all(path is not None for path in chessfly_paths):
        raise SystemExit("--chessfly-connectome, --chessfly-neurons, and --chessfly-weights are required together")
    if any(path is not None for path in flynet_paths) and not all(path is not None for path in flynet_paths):
        raise SystemExit("--flynet-weights and --flynet-graph are required together")
    if args.flynet_graph_metadata is not None and not all(path is not None for path in flynet_paths):
        raise SystemExit("--flynet-graph-metadata requires --flynet-weights and --flynet-graph")
    if args.flynet_config is not None and not all(path is not None for path in flynet_paths):
        raise SystemExit("--flynet-config requires --flynet-weights and --flynet-graph")
    if all(path is not None for path in flynet_paths):
        from .flynet import FlyNetPolicy
        from .flynet_training import load_flynet_model

        fly = FlyNetPolicy(
            load_flynet_model(
                args.flynet_weights,
                args.flynet_graph,
                graph_metadata_path=args.flynet_graph_metadata,
                config_path=args.flynet_config,
                device=args.flynet_device,
            )
        )
        policy_name = "flynet"
    elif all(path is not None for path in chessfly_paths):
        from .chessfly import ChessFlyModel, ChessFlyPolicy

        fly = ChessFlyPolicy(
            ChessFlyModel.from_artifacts(
                args.chessfly_connectome,
                args.chessfly_neurons,
                args.chessfly_weights,
                device=args.chessfly_device,
            )
        )
        policy_name = "chessfly"
    elif args.connectome is None:
        fly = SurrogateFlyBrain(seed=args.seed)
        policy_name = "surrogate"
    else:
        fly = ConnectomeFlyBrain(
            load_connectome(args.connectome),
            steps_per_position=args.neural_steps,
        )
        policy_name = "connectome"

    with StockfishEngine(args.engine, depth=args.depth) as stockfish:
        result = play_game(
            fly,
            stockfish,
            fly_color=fly_color,
            board=board,
            max_plies=args.max_plies,
        )
        engine_settings = {
            "path": stockfish.path,
            "depth": stockfish.depth,
            "threads": 1,
            "hash_mb": 16,
            "fly_policy": policy_name,
        }
        if args.connectome is not None:
            engine_settings["connectome"] = str(args.connectome)
        if all(path is not None for path in chessfly_paths):
            engine_settings.update(
                {
                    "chessfly_connectome": str(args.chessfly_connectome),
                    "chessfly_neurons": str(args.chessfly_neurons),
                    "chessfly_weights": str(args.chessfly_weights),
                    "chessfly_device": args.chessfly_device,
                }
            )
        if all(path is not None for path in flynet_paths):
            engine_settings.update(
                {
                    "flynet_weights": str(args.flynet_weights),
                    "flynet_graph": str(args.flynet_graph),
                    "flynet_graph_metadata": str(args.flynet_graph_metadata)
                    if args.flynet_graph_metadata is not None
                    else None,
                    "flynet_config": str(args.flynet_config) if args.flynet_config is not None else None,
                    "flynet_device": args.flynet_device,
                }
            )

    if args.record is not None:
        write_experiment(
            args.record,
            result,
            seed=args.seed,
            engine_settings=engine_settings,
            result_path=args.result,
        )

    if not args.quiet:
        for record in result.moves:
            print(f"{record.ply:>3}  {record.actor:<9} {record.san:<8} ({record.uci})")
    outcome = result.outcome
    print(f"result={outcome.result() if outcome else '*'} plies={len(result.moves)} fen={result.board.fen()}")
    if args.record is not None:
        print(f"record={args.record}")
        if args.result is not None:
            print(f"result_json={args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
