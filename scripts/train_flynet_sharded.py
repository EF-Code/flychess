#!/usr/bin/env python3
"""Train FlyNet from a resumable sharded dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flychess.flynet_scale import train_flynet_sharded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream a large FlyNet dataset through a scratch model.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/flynet-large-release"))
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=2_048)
    parser.add_argument("--edges", type=int, default=64_000)
    parser.add_argument("--input-nodes", type=int, default=256)
    parser.add_argument("--readout-nodes", type=int, default=320)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--readout-width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--validation-limit", type=int, default=100_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = train_flynet_sharded(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        graph_seed=args.graph_seed,
        node_count=args.nodes,
        edge_count=args.edges,
        input_nodes=args.input_nodes,
        readout_nodes=args.readout_nodes,
        steps=args.steps,
        readout_width=args.readout_width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        value_weight=args.value_weight,
        validation_fraction=args.validation_fraction,
        validation_limit=args.validation_limit,
        device=args.device,
        resume=not args.no_resume,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "release": result["release"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
