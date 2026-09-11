import json
from pathlib import Path

import chess
import pytest

from flychess.experiment import (
    IllegalMoveError,
    ExperimentFormatError,
    ExperimentRecorder,
    ReplayValidationError,
    atomic_write_json,
    read_experiment,
    write_experiment,
)
from flychess.game import GameResult, MoveRecord


def _game_result() -> GameResult:
    board = chess.Board()
    records: list[MoveRecord] = []
    for actor, uci in (
        ("fly", "e2e4"),
        ("stockfish", "e7e5"),
        ("fly", "g1f3"),
        ("stockfish", "b8c6"),
    ):
        fen_before = board.fen()
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        board.push(move)
        records.append(
            MoveRecord(
                ply=len(records) + 1,
                actor=actor,
                san=san,
                uci=uci,
                fen_before=fen_before,
                fen_after=board.fen(),
            )
        )
    return GameResult(board=board, moves=records, max_plies=4)


def _write_lines(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_write_and_replay_preserves_reproduction_metadata_and_state(tmp_path: Path) -> None:
    log_path = tmp_path / "game.jsonl"
    result_path = tmp_path / "result.json"
    recorded = write_experiment(
        log_path,
        _game_result(),
        seed=41,
        engine_settings={"path": "/usr/bin/stockfish", "depth": 3, "threads": 1},
        experiment_id="test-run",
        created_at="2026-09-11T00:00:00Z",
        finished_at="2026-09-11T00:00:01Z",
        timestamp_factory=lambda: "2026-09-11T00:00:00Z",
        result_path=result_path,
    )

    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [line["type"] for line in lines] == ["metadata", "move", "move", "move", "move", "result"]
    assert [line["seq"] for line in lines] == list(range(6))
    assert lines[0]["start_fen"] == chess.Board().fen()
    assert lines[0]["seed"] == 41
    assert lines[0]["engine_settings"]["depth"] == 3
    assert all({"fen_before", "fen_after", "actor", "uci", "san", "timestamp"} <= line.keys() for line in lines[1:-1])
    assert lines[-1]["outcome"]["result"] == "*"
    assert lines[-1]["final_fen"] == recorded.board.fen()

    replayed = read_experiment(log_path)
    assert replayed.board.fen() == _game_result().board.fen()
    assert replayed.final_fen == replayed.board.fen()
    assert replayed.result_marker == "*"
    assert replayed.move_count == 4
    assert json.loads(result_path.read_text(encoding="utf-8"))["final_fen"] == replayed.final_fen


def test_atomic_json_write_replaces_only_after_serialization_succeeds(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    atomic_write_json(path, {"version": 1, "value": "first"})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == "first"

    with pytest.raises(ValueError):
        atomic_write_json(path, {"value": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == "first"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []

    atomic_write_json(path, {"version": 2, "value": "second"})
    assert json.loads(path.read_text(encoding="utf-8"))["value"] == "second"


def test_recorder_persists_incremental_moves_and_finishes_replayable_log(tmp_path: Path) -> None:
    log_path = tmp_path / "incremental.jsonl"
    recorder = ExperimentRecorder(
        log_path,
        seed=7,
        engine_settings={"depth": 1},
        max_plies=2,
        experiment_id="incremental",
        created_at="2026-09-11T00:00:00Z",
        timestamp_factory=lambda: "2026-09-11T00:00:00Z",
    )
    recorder.record_move("e2e4")
    assert '"type":"move"' in log_path.read_text(encoding="utf-8")
    with pytest.raises(ExperimentFormatError):
        read_experiment(log_path)
    recorder.record_move(chess.Move.from_uci("e7e5"), actor="stockfish")
    replayed = recorder.finish(stopped_at_limit=True, finished_at="2026-09-11T00:00:01Z")
    assert replayed.move_count == 2
    assert replayed.result["stopped_at_limit"] is True
    assert replayed.board.fen() == recorder.board.fen()

    with pytest.raises(ValueError, match="already finished"):
        recorder.finish()


def test_reader_rejects_malformed_json_duplicate_keys_and_unknown_events(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"type":"metadata","type":"move"}\n', encoding="utf-8")
    with pytest.raises(ExperimentFormatError, match="duplicate JSON key"):
        read_experiment(path)

    _write_lines(path, [{"type": "metadata", "schema": "wrong"}, {"type": "result"}])
    with pytest.raises(ExperimentFormatError, match="unsupported schema"):
        read_experiment(path)

    _write_lines(path, [{"type": "metadata", "schema": "flychess.experiment/v1", "seq": 0}, {"type": "telemetry"}])
    with pytest.raises(ExperimentFormatError, match="missing required field"):
        read_experiment(path)


def test_reader_rejects_illegal_moves_and_state_tampering(tmp_path: Path) -> None:
    log_path = tmp_path / "game.jsonl"
    write_experiment(log_path, _game_result(), seed=1, engine_settings={"depth": 1}, timestamp_factory=lambda: "t0")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    illegal = [dict(record) for record in records]
    illegal[1]["uci"] = "e2e5"
    _write_lines(log_path, illegal)
    with pytest.raises(IllegalMoveError):
        read_experiment(log_path)

    fen_tampered = [dict(record) for record in records]
    fen_tampered[2]["fen_before"] = fen_tampered[1]["fen_before"]
    _write_lines(log_path, fen_tampered)
    with pytest.raises(ReplayValidationError, match="fen_before"):
        read_experiment(log_path)

    actor_tampered = [dict(record) for record in records]
    actor_tampered[1]["actor"] = "stockfish"
    _write_lines(log_path, actor_tampered)
    with pytest.raises(ReplayValidationError, match="actor"):
        read_experiment(log_path)


def test_reader_rejects_san_sequence_and_final_outcome_tampering(tmp_path: Path) -> None:
    log_path = tmp_path / "game.jsonl"
    write_experiment(log_path, _game_result(), seed=1, engine_settings={"depth": 1}, timestamp_factory=lambda: "t0")
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    san_tampered = [dict(record) for record in records]
    san_tampered[1]["san"] = "d4"
    _write_lines(log_path, san_tampered)
    with pytest.raises(ReplayValidationError, match="SAN"):
        read_experiment(log_path)

    outcome_tampered = [dict(record) for record in records]
    outcome_tampered[-1]["outcome"] = {
        "result": "1-0",
        "winner": "white",
        "termination": "checkmate",
    }
    outcome_tampered[-1]["result"] = "1-0"
    _write_lines(log_path, outcome_tampered)
    with pytest.raises(ReplayValidationError, match="recorded result"):
        read_experiment(log_path)
