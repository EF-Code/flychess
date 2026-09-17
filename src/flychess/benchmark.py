"""Reproducible Flychess-policy-vs-Stockfish benchmark helpers."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import chess

from .chessfly import ChessFlyModel, ChessFlyPolicy
from .engine import StockfishEngine


DEFAULT_BENCHMARK_FENS: tuple[str, ...] = (
    chess.Board().fen(),
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqk2r/pppp1ppp/2n2n2/8/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3",
    "r3k2r/ppp2ppp/2n1b3/3qp3/3P4/2P1PN2/PP3PPP/R2Q1RK1 w kq - 4 12",
    "8/5pk1/3p2p1/1p2p2p/1P2P2P/P2P2P1/5PK1/8 w - - 0 1",
)


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    index: int
    fen: str
    engine_move: str
    fly_move: str
    legal: bool
    engine_top1: bool
    engine_top5: bool
    latency_ms: float
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "fen": self.fen,
            "engine_move": self.engine_move,
            "fly_move": self.fly_move,
            "legal": self.legal,
            "engine_top1": self.engine_top1,
            "engine_top5": self.engine_top5,
            "latency_ms": round(self.latency_ms, 3),
            "candidate_count": self.candidate_count,
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_chessfly(
    model: ChessFlyModel,
    *,
    engine_path: str = "stockfish",
    depth: int = 2,
    fens: Sequence[str] = DEFAULT_BENCHMARK_FENS,
) -> dict[str, Any]:
    """Compare the legacy compatibility policy with a fixed-depth reference."""
    return benchmark_policy(
        ChessFlyPolicy(model),
        label="legacy-comparison",
        model_metadata=model.graph.summary(),
        engine_path=engine_path,
        depth=depth,
        fens=fens,
    )


def benchmark_flynet(
    model: Any,
    *,
    engine_path: str = "stockfish",
    depth: int = 2,
    fens: Sequence[str] = DEFAULT_BENCHMARK_FENS,
) -> dict[str, Any]:
    """Compare an independent FlyNet policy with a fixed-depth reference."""

    from .flynet import FlyNetPolicy

    return benchmark_policy(
        FlyNetPolicy(model),
        label="flynet",
        model_metadata=model.graph.summary(),
        engine_path=engine_path,
        depth=depth,
        fens=fens,
    )


def benchmark_policy(
    policy: Any,
    *,
    label: str,
    model_metadata: Mapping[str, Any] | None = None,
    engine_path: str = "stockfish",
    depth: int = 2,
    fens: Sequence[str] = DEFAULT_BENCHMARK_FENS,
) -> dict[str, Any]:
    """Benchmark any legal-move policy against one fixed engine configuration."""

    rows: list[BenchmarkRow] = []
    with StockfishEngine(path=engine_path, depth=depth) as engine:
        for index, fen in enumerate(fens):
            board = chess.Board(fen)
            engine_moves = engine.top_moves(board, count=5)
            engine_move = engine_moves[0] if engine_moves else engine.choose_move(board)
            started = time.perf_counter()
            fly_move = policy.select_move(board)
            latency_ms = (time.perf_counter() - started) * 1000.0
            candidates = policy.last_readout.candidates if policy.last_readout is not None else ()
            top5 = {candidate.uci for candidate in candidates[:5]}
            rows.append(
                BenchmarkRow(
                    index=index,
                    fen=fen,
                    engine_move=engine_move.uci(),
                    fly_move=fly_move.uci(),
                    legal=fly_move in board.legal_moves,
                    engine_top1=fly_move == engine_move,
                    engine_top5=engine_move.uci() in top5,
                    latency_ms=latency_ms,
                    candidate_count=len(candidates),
                )
            )
    latencies = [row.latency_ms for row in rows]
    total = len(rows)
    return {
        "policy": label,
        "engine": {"path": engine_path, "depth": depth},
        "model": dict(model_metadata or {}),
        "positions": total,
        "legal_rate": sum(row.legal for row in rows) / total if total else 0.0,
        "engine_top1_rate": sum(row.engine_top1 for row in rows) / total if total else 0.0,
        "engine_top5_rate": sum(row.engine_top5 for row in rows) / total if total else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
        },
        "rows": [row.to_dict() for row in rows],
    }


def compare_policies(
    policies: Mapping[str, Any],
    *,
    engine_path: str = "stockfish",
    depth: int = 2,
    fens: Sequence[str] = DEFAULT_BENCHMARK_FENS,
) -> dict[str, Any]:
    """Run named policies on the same FEN suite for apples-to-apples reports."""

    reports: dict[str, Any] = {}
    for label, policy in policies.items():
        model_metadata = getattr(getattr(policy, "model", None), "graph", None)
        metadata = model_metadata.summary() if model_metadata is not None else {}
        reports[label] = benchmark_policy(
            policy,
            label=label,
            model_metadata=metadata,
            engine_path=engine_path,
            depth=depth,
            fens=fens,
        )
    return {
        "engine": {"path": engine_path, "depth": depth},
        "positions": len(fens),
        "policies": reports,
    }


__all__ = [
    "BenchmarkRow",
    "DEFAULT_BENCHMARK_FENS",
    "benchmark_chessfly",
    "benchmark_flynet",
    "benchmark_policy",
    "compare_policies",
]
