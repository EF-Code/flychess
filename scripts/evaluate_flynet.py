#!/usr/bin/env python3
"""Evaluate an independent FlyNet release on a held-out labeled dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess

from flychess.flynet import action_index, encode_flynet_board
from flychess.flynet_training import load_dataset, load_flynet_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate FlyNet legal-move and teacher agreement metrics.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--graph-metadata", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N rows; 0 means all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(args.dataset)
    model = load_flynet_model(
        args.weights,
        args.graph,
        graph_metadata_path=args.graph_metadata,
        config_path=args.config,
        device=args.device,
    )
    rows = dataset.sample_count if args.limit <= 0 else min(args.limit, dataset.sample_count)
    top1 = 0
    top5 = 0
    legal = 0
    latencies: list[float] = []
    for index in range(rows):
        board = chess.Board(dataset.fens[index])
        started = time.perf_counter()
        output = model.forward([encode_flynet_board(board)])
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        logits = output.policy_logits[0].detach().cpu()
        legal_indices = [(move, action_index(move)) for move in board.legal_moves]
        ranked = sorted(legal_indices, key=lambda pair: float(logits[pair[1]]), reverse=True)
        selected = ranked[0][0]
        legal += int(selected in board.legal_moves)
        top1 += int(ranked[0][1] == int(dataset.action_indices[index]))
        top5 += int(any(action == int(dataset.action_indices[index]) for _move, action in ranked[:5]))
    result = {
        "model": "FlyNet",
        "samples": rows,
        "legal_rate": legal / rows if rows else 0.0,
        "teacher_top1_rate": top1 / rows if rows else 0.0,
        "teacher_top5_rate": top5 / rows if rows else 0.0,
        "latency_ms": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
