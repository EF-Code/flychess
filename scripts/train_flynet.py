#!/usr/bin/env python3
"""Train and package an independent FlyNet release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flychess.flynet_training import load_dataset, train_flynet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FlyNet from random initialization.")
    parser.add_argument("--dataset", type=Path, default=Path("/content/flynet-dataset.npz"))
    parser.add_argument("--dataset-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/flynet-release"))
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument("--nodes", type=int, default=2_048)
    parser.add_argument("--edges", type=int, default=64_000)
    parser.add_argument("--input-nodes", type=int, default=256)
    parser.add_argument("--readout-nodes", type=int, default=320)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--readout-width", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--device", default=None, help="torch device; defaults to cuda when available")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.dataset, args.dataset_metadata)
    result = train_flynet(
        dataset,
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
        device=args.device,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "release": result["release"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
