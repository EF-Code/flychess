#!/usr/bin/env python3
"""Generate the independent FlyNet supervised dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flychess.flynet_training import generate_teacher_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate legal FlyNet positions with Stockfish labels.")
    parser.add_argument("--engine", default="stockfish", help="Stockfish executable path")
    parser.add_argument("--depth", type=int, default=3, help="Stockfish labeling depth")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--min-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("/content/flynet-dataset.npz"))
    parser.add_argument("--metadata", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = generate_teacher_dataset(
        engine_path=args.engine,
        depth=args.depth,
        count=args.samples,
        seed=args.seed,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        output_path=args.output,
        metadata_path=args.metadata,
    )
    print(json.dumps({"dataset": str(args.output), "samples": dataset.sample_count, **dataset.metadata}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
