#!/usr/bin/env python3
"""Generate a resumable sharded Stockfish-labelled FlyNet dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flychess.flynet_scale import generate_teacher_shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate large FlyNet datasets in resumable compressed shards.")
    parser.add_argument("--root", type=Path, default=Path("/content/flynet-shards"))
    parser.add_argument("--engine", default="stockfish")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--samples", type=int, default=50_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--min-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(record: dict[str, object], complete: int, total: int) -> None:
        if not args.quiet:
            print(
                json.dumps(
                    {
                        "shard": record["shard_index"],
                        "samples": record["samples"],
                        "completed_shards": complete,
                        "shard_count": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = generate_teacher_shards(
        root=args.root,
        engine_path=args.engine,
        depth=args.depth,
        samples=args.samples,
        shard_size=args.shard_size,
        seed=args.seed,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        workers=args.workers,
        resume=not args.no_resume,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "dataset_root": str(args.root),
                "complete": manifest["complete"],
                "samples": manifest["completed_samples"],
                "shards": manifest["completed_shards"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
