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
        description="Play a bounded chess game between the fly-brain baseline and Stockfish.",
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
    if args.connectome is None:
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
