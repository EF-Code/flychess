#!/usr/bin/env python3
"""Run the fixed FlyNet-vs-Stockfish policy benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flychess.benchmark import DEFAULT_BENCHMARK_FENS, benchmark_flynet
from flychess.flynet_training import load_flynet_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a FlyNet release on a fixed FEN suite.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--fens", type=Path, default=None, help="Text file with one FEN per line")
    parser.add_argument("--device", default="cpu")
    return parser


def _load_fens(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_BENCHMARK_FENS
    fens = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))
    if not fens:
        raise ValueError("FEN file does not contain any positions")
    return fens


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = load_flynet_model(
        args.weights,
        args.graph,
        graph_metadata_path=args.graph_metadata,
        config_path=args.config,
        device=args.device,
    )
    report = benchmark_flynet(
        model,
        engine_path=args.engine,
        depth=args.depth,
        fens=_load_fens(args.fens),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
