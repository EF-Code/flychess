"""Persistent recording and validation for Flychess experiments.

The experiment log is newline-delimited JSON (JSONL).  A complete log has
exactly this shape:

* one ``metadata`` record with ``seq == 0``;
* zero or more ``move`` records with contiguous sequence numbers; and
* one final ``result`` record.

The records intentionally contain redundant state.  A move stores the FEN
before and after it, its actor, UCI, SAN, and a timestamp.  The monotonic
``seq`` field is the ordering authority, so replay does not depend on wall
clock precision.  Metadata stores the starting FEN, fly colour, policy seed,
engine settings, and ply limit.  The result stores the final FEN and a
structured chess outcome.

Example records (whitespace omitted here for readability)::

    {"type":"metadata","schema":"flychess.experiment/v1","seq":0,
     "experiment_id":"...","created_at":"2026-01-01T00:00:00Z",
     "start_fen":"...","fly_color":"white","seed":17,
     "engine_settings":{"path":"stockfish","depth":4},"max_plies":80}
    {"type":"move","seq":1,"ply":1,"timestamp":"...",
     "actor":"fly","fen_before":"...","uci":"e2e4","san":"e4",
     "fen_after":"..."}
    {"type":"result","seq":2,"finished_at":"...","plies":1,
     "final_fen":"...","outcome":{"result":"*","winner":null,
     "termination":null},"stopped_at_limit":false}

``write_experiment`` writes a complete game log atomically.  ``write_result``
and ``atomic_write_json`` provide the same guarantee for a single JSON result
file.  ``read_experiment`` validates every record while replaying it, and
raises a ``ReplayValidationError`` (a ``ValueError``) for malformed logs,
state mismatches, or illegal chess moves.

Only the standard library and the existing ``python-chess`` dependency are
used here.  The module does not modify the game, engine, brain, or CLI APIs.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess


SCHEMA = "flychess.experiment/v1"
"""Schema identifier written to the metadata record."""

METADATA = "metadata"
MOVE = "move"
RESULT = "result"
_EVENT_TYPES = frozenset((METADATA, MOVE, RESULT))
_OUTCOME_RESULTS = frozenset(("1-0", "0-1", "1/2-1/2", "*"))
_ACTORS = frozenset(("fly", "stockfish"))


class ExperimentError(ValueError):
    """Base class for invalid experiment data or persistence inputs."""


class ExperimentFormatError(ExperimentError):
    """Raised when an experiment file is not valid Flychess JSONL."""


class ReplayValidationError(ExperimentFormatError):
    """Raised when a valid-looking log cannot be replayed faithfully."""


class IllegalMoveError(ReplayValidationError):
    """Raised when a move in a log is not legal in the preceding position."""


@dataclass(frozen=True)
class ReplayedExperiment:
    """Validated metadata, moves, result, and the reconstructed final board.

    The dictionaries contain the JSON records as read from disk.  ``board``
    is a fresh ``python-chess`` board reconstructed from ``start_fen`` and
    every recorded UCI move; callers can inspect it with normal chess APIs.
    """

    metadata: Mapping[str, Any]
    moves: tuple[Mapping[str, Any], ...]
    result: Mapping[str, Any]
    board: chess.Board = field(repr=False, compare=False)

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        """Return all records in their on-disk order."""

        return (self.metadata, *self.moves, self.result)

    @property
    def final_fen(self) -> str:
        """The final FEN recorded in the result record."""

        return str(self.result["final_fen"])

    @property
    def outcome(self) -> Mapping[str, Any]:
        """The structured outcome from the result record."""

        outcome = self.result["outcome"]
        if not isinstance(outcome, Mapping):  # pragma: no cover - validated
            raise RuntimeError("validated outcome record is not a mapping")
        return outcome

    @property
    def result_marker(self) -> str:
        """The PGN-style result marker (``1-0``, ``0-1``, ``1/2-1/2``, ``*``)."""

        return str(self.outcome["result"])

    @property
    def move_count(self) -> int:
        """Number of replayed plies."""

        return len(self.moves)


# These aliases make the return type discoverable under the two terms most
# users naturally reach for when integrating an experiment recorder.
Experiment = ReplayedExperiment
ReplayResult = ReplayedExperiment


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def _load_json_line(line: str, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise ExperimentFormatError(f"line {line_number}: blank JSONL records are not allowed")
    try:
        value = json.loads(
            line,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExperimentFormatError(f"line {line_number}: malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentFormatError(f"line {line_number}: event must be a JSON object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require(record: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in record:
        raise ExperimentFormatError(f"{context}: missing required field {key!r}")
    return record[key]


def _require_string(value: Any, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentFormatError(f"{context}: {field_name!r} must be a non-empty string")
    return value


def _require_sequence(value: Any, field_name: str, context: str, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ExperimentFormatError(f"{context}: {field_name!r} must be an integer >= {minimum}")
    return value


def _validate_json_value(value: Any, context: str) -> None:
    """Reject values that would be lossy or invalid in strict JSON."""

    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentFormatError(f"{context}: non-finite numbers are not allowed")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ExperimentFormatError(f"{context}: object keys must be strings")
            _validate_json_value(nested, f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json_value(nested, f"{context}[{index}]")
        return
    raise ExperimentFormatError(f"{context}: value of type {type(value).__name__} is not JSON-safe")


def _copy_json_mapping(value: Any, field_name: str, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentFormatError(f"{context}: {field_name!r} must be a JSON object")
    _validate_json_value(value, f"{context}.{field_name}")
    # Round-tripping makes the stored mapping independent of caller-owned
    # nested dictionaries and converts tuples to normal JSON arrays.
    try:
        copied = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise ExperimentFormatError(f"{context}: {field_name!r} is not JSON-safe") from exc
    if not isinstance(copied, dict):  # pragma: no cover - Mapping guard above
        raise ExperimentFormatError(f"{context}: {field_name!r} must be a JSON object")
    return copied


def _board_from_fen(fen: Any, field_name: str, context: str) -> chess.Board:
    fen_text = _require_string(fen, field_name, context)
    try:
        board = chess.Board(fen_text)
    except ValueError as exc:
        raise ExperimentFormatError(f"{context}: {field_name!r} is not valid FEN: {exc}") from exc
    if not board.is_valid():
        raise ExperimentFormatError(f"{context}: {field_name!r} describes an invalid chess position")
    return board


def _validate_metadata(record: Mapping[str, Any]) -> chess.Board:
    context = "metadata record"
    if record.get("type") != METADATA:
        raise ExperimentFormatError(f"{context}: first record must have type {METADATA!r}")
    if record.get("schema") != SCHEMA:
        raise ExperimentFormatError(f"{context}: unsupported schema {record.get('schema')!r}")
    if record.get("seq") != 0:
        raise ExperimentFormatError(f"{context}: seq must be 0")

    _require_string(_require(record, "experiment_id", context), "experiment_id", context)
    _require_string(_require(record, "created_at", context), "created_at", context)
    start_board = _board_from_fen(_require(record, "start_fen", context), "start_fen", context)
    fly_color = _require(record, "fly_color", context)
    if fly_color not in ("white", "black"):
        raise ExperimentFormatError(f"{context}: fly_color must be 'white' or 'black'")

    seed = _require(record, "seed", context)
    if not _is_int(seed):
        raise ExperimentFormatError(f"{context}: seed must be an integer")
    _copy_json_mapping(_require(record, "engine_settings", context), "engine_settings", context)
    _require_sequence(_require(record, "max_plies", context), "max_plies", context, minimum=1)
    return start_board


def _expected_actor(board: chess.Board, fly_color: str) -> str:
    fly_turn = (board.turn == chess.WHITE and fly_color == "white") or (
        board.turn == chess.BLACK and fly_color == "black"
    )
    return "fly" if fly_turn else "stockfish"


def _outcome_details(board: chess.Board) -> dict[str, Any]:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return {"result": "*", "winner": None, "termination": None}
    winner = None if outcome.winner is None else ("white" if outcome.winner else "black")
    termination = outcome.termination.name.lower()
    return {"result": outcome.result(), "winner": winner, "termination": termination}


def _validate_outcome(record: Mapping[str, Any], board: chess.Board, context: str) -> dict[str, Any]:
    raw_outcome = _require(record, "outcome", context)
    if not isinstance(raw_outcome, Mapping):
        raise ExperimentFormatError(f"{context}: outcome must be a JSON object")
    outcome = dict(raw_outcome)
    actual = _outcome_details(board)
    for key in ("result", "winner", "termination"):
        if key not in outcome:
            raise ExperimentFormatError(f"{context}: outcome is missing {key!r}")
    if outcome["result"] not in _OUTCOME_RESULTS:
        raise ExperimentFormatError(f"{context}: outcome.result is not a chess result marker")
    if outcome != {"result": outcome["result"], "winner": outcome["winner"], "termination": outcome["termination"]}:
        # Unknown nested outcome fields are not harmful, but validate the
        # canonical values below.  This branch intentionally does nothing.
        pass
    if outcome["result"] != actual["result"]:
        raise ReplayValidationError(
            f"{context}: recorded result {outcome['result']!r} does not match replayed result {actual['result']!r}"
        )
    if outcome["winner"] != actual["winner"]:
        raise ReplayValidationError(
            f"{context}: recorded winner {outcome['winner']!r} does not match replayed winner {actual['winner']!r}"
        )
    if outcome["termination"] != actual["termination"]:
        raise ReplayValidationError(
            f"{context}: recorded termination {outcome['termination']!r} does not match replayed termination {actual['termination']!r}"
        )
    if "result" in record and record["result"] != outcome["result"]:
        raise ReplayValidationError(f"{context}: result alias does not match outcome.result")
    return outcome


def _validate_records(records: list[dict[str, Any]]) -> ReplayedExperiment:
    if len(records) < 2:
        raise ExperimentFormatError("experiment must contain metadata and result records")
    metadata = records[0]
    board = _validate_metadata(metadata)
    moves: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    expected_seq = 1
    fly_color = str(metadata["fly_color"])
    max_plies = int(metadata["max_plies"])

    for index, record in enumerate(records[1:], start=2):
        event_type = record.get("type")
        context = f"line {index} ({event_type or 'unknown'} record)"
        if event_type == METADATA:
            raise ExperimentFormatError(f"{context}: metadata is only allowed as the first record")
        if event_type == MOVE:
            if result is not None:
                raise ExperimentFormatError(f"{context}: move appears after result record")
            if record.get("seq") != expected_seq:
                raise ExperimentFormatError(f"{context}: seq must be {expected_seq}")
            ply = _require_sequence(_require(record, "ply", context), "ply", context, minimum=1)
            if ply != len(moves) + 1:
                raise ReplayValidationError(f"{context}: ply must be {len(moves) + 1}")
            if ply > max_plies:
                raise ReplayValidationError(f"{context}: move exceeds metadata max_plies={max_plies}")

            actor = _require(record, "actor", context)
            if actor not in _ACTORS:
                raise ExperimentFormatError(f"{context}: actor must be 'fly' or 'stockfish'")
            expected_actor = _expected_actor(board, fly_color)
            if actor != expected_actor:
                raise ReplayValidationError(
                    f"{context}: actor {actor!r} is invalid; expected {expected_actor!r} for this turn"
                )
            if "timestamp" in record:
                _require_string(record["timestamp"], "timestamp", context)

            fen_before = _require_string(_require(record, "fen_before", context), "fen_before", context)
            if fen_before != board.fen():
                raise ReplayValidationError(
                    f"{context}: fen_before does not match the position at ply {ply}"
                )
            uci = _require_string(_require(record, "uci", context), "uci", context)
            try:
                move = chess.Move.from_uci(uci)
            except ValueError as exc:
                raise ExperimentFormatError(f"{context}: invalid UCI move {uci!r}") from exc
            if move not in board.legal_moves:
                raise IllegalMoveError(f"{context}: illegal move {uci!r} for recorded FEN")
            san = _require_string(_require(record, "san", context), "san", context)
            expected_san = board.san(move)
            if san != expected_san:
                raise ReplayValidationError(
                    f"{context}: SAN {san!r} does not match UCI {uci!r}; expected {expected_san!r}"
                )
            board.push(move)
            fen_after = _require_string(_require(record, "fen_after", context), "fen_after", context)
            if fen_after != board.fen():
                raise ReplayValidationError(f"{context}: fen_after does not match the replayed position")
            moves.append(dict(record))
            expected_seq += 1
            continue

        if event_type == RESULT:
            if result is not None:
                raise ExperimentFormatError(f"{context}: duplicate result record")
            if index != len(records):
                raise ExperimentFormatError(f"{context}: result record must be last")
            if record.get("seq") != expected_seq:
                raise ExperimentFormatError(f"{context}: seq must be {expected_seq}")
            plies = _require_sequence(_require(record, "plies", context), "plies", context, minimum=0)
            if plies != len(moves):
                raise ReplayValidationError(f"{context}: plies does not match the number of move records")
            final_fen = _require_string(_require(record, "final_fen", context), "final_fen", context)
            if final_fen != board.fen():
                raise ReplayValidationError(f"{context}: final_fen does not match the replayed position")
            _validate_outcome(record, board, context)
            stopped_at_limit = _require(record, "stopped_at_limit", context)
            if not isinstance(stopped_at_limit, bool):
                raise ExperimentFormatError(f"{context}: stopped_at_limit must be boolean")
            if "finished_at" in record:
                _require_string(record["finished_at"], "finished_at", context)
            result = dict(record)
            expected_seq += 1
            continue

        if event_type not in _EVENT_TYPES:
            raise ExperimentFormatError(f"{context}: unknown event type {event_type!r}")

    if result is None:
        raise ExperimentFormatError("experiment must end with one result record")
    return ReplayedExperiment(
        metadata=dict(metadata),
        moves=tuple(moves),
        result=result,
        board=board,
    )


def _read_records(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentError(f"unable to read experiment log {source}: {exc}") from exc
    if not text:
        raise ExperimentFormatError(f"experiment log {source} is empty")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        records.append(_load_json_line(line, line_number))
    return records


def read_experiment(path: str | os.PathLike[str]) -> ReplayedExperiment:
    """Read, replay, and validate a complete JSONL experiment log."""

    return _validate_records(_read_records(path))


def replay_experiment(path: str | os.PathLike[str]) -> ReplayedExperiment:
    """Alias for :func:`read_experiment` emphasizing the replay operation."""

    return read_experiment(path)


def validate_experiment(path: str | os.PathLike[str]) -> ReplayedExperiment:
    """Validate a log and return its replay, without modifying the file."""

    return read_experiment(path)


def _timestamp(now: Callable[[], Any] | None = None) -> str:
    value = now() if now is not None else datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str) and value:
        return value
    raise ExperimentError("timestamp factory must return a non-empty string or datetime")


def _color_name(color: chess.Color) -> str:
    if color is chess.WHITE or color is True:
        return "white"
    if color is chess.BLACK or color is False:
        return "black"
    raise ExperimentError("fly_color must be chess.WHITE/True or chess.BLACK/False")


def _make_metadata(
    game_result: Any,
    *,
    seed: int,
    engine_settings: Mapping[str, Any],
    experiment_id: str | None,
    created_at: str | None,
) -> dict[str, Any]:
    context = "experiment metadata"
    if not _is_int(seed):
        raise ExperimentError("seed must be an integer")
    settings = _copy_json_mapping(engine_settings, "engine_settings", context)
    start_fen = getattr(game_result, "start_fen", None)
    start_board = _board_from_fen(start_fen, "start_fen", context)
    max_plies = getattr(game_result, "max_plies", None)
    _require_sequence(max_plies, "max_plies", context, minimum=1)
    fly_color = _color_name(getattr(game_result, "fly_color", None))
    return {
        "type": METADATA,
        "schema": SCHEMA,
        "seq": 0,
        "experiment_id": experiment_id or uuid.uuid4().hex,
        "created_at": created_at or _timestamp(),
        "start_fen": start_board.fen(),
        "fly_color": fly_color,
        "seed": seed,
        "engine_settings": settings,
        "max_plies": max_plies,
    }


def _make_result_record(
    board: chess.Board,
    move_count: int,
    *,
    seq: int,
    stopped_at_limit: bool,
    finished_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(stopped_at_limit, bool):
        raise ExperimentError("stopped_at_limit must be boolean")
    outcome = _outcome_details(board)
    return {
        "type": RESULT,
        "seq": seq,
        "finished_at": finished_at or _timestamp(),
        "plies": move_count,
        "final_fen": board.fen(),
        "outcome": outcome,
        # The top-level alias is convenient for simple line-oriented tools;
        # replay validation requires it to agree with outcome.result if used.
        "result": outcome["result"],
        "stopped_at_limit": stopped_at_limit,
    }


def _records_from_game_result(
    game_result: Any,
    *,
    seed: int,
    engine_settings: Mapping[str, Any],
    experiment_id: str | None,
    created_at: str | None,
    finished_at: str | None,
    timestamp_factory: Callable[[], Any] | None,
) -> list[dict[str, Any]]:
    metadata = _make_metadata(
        game_result,
        seed=seed,
        engine_settings=engine_settings,
        experiment_id=experiment_id,
        created_at=created_at,
    )
    board = chess.Board(metadata["start_fen"])
    records: list[dict[str, Any]] = [metadata]
    game_moves = getattr(game_result, "moves", None)
    if not isinstance(game_moves, Iterable):
        raise ExperimentError("game_result.moves must be iterable")
    fly_color = metadata["fly_color"]
    for expected_ply, move_record in enumerate(game_moves, start=1):
        context = f"game move {expected_ply}"
        actor = getattr(move_record, "actor", None)
        if actor not in _ACTORS:
            raise ExperimentError(f"{context}: actor must be 'fly' or 'stockfish'")
        expected_actor = _expected_actor(board, fly_color)
        if actor != expected_actor:
            raise ExperimentError(f"{context}: actor {actor!r} is invalid; expected {expected_actor!r}")
        record = {
            "type": MOVE,
            "seq": expected_ply,
            "ply": getattr(move_record, "ply", expected_ply),
            "timestamp": _timestamp(timestamp_factory),
            "actor": actor,
            "fen_before": getattr(move_record, "fen_before", None),
            "uci": getattr(move_record, "uci", None),
            "san": getattr(move_record, "san", None),
            "fen_after": getattr(move_record, "fen_after", None),
        }
        records.append(record)
        # Validate the supplied game record's state and advance our own board.
        # Calling the same validator later also checks all result fields.
        try:
            move = chess.Move.from_uci(record["uci"])
        except (TypeError, ValueError) as exc:
            raise ExperimentError(f"{context}: invalid UCI move") from exc
        if record["fen_before"] != board.fen():
            raise ExperimentError(f"{context}: fen_before does not match the game start state")
        if move not in board.legal_moves:
            raise ExperimentError(f"{context}: game result contains illegal move {record['uci']!r}")
        if record["san"] != board.san(move):
            raise ExperimentError(f"{context}: SAN does not match UCI")
        board.push(move)
        if record["fen_after"] != board.fen():
            raise ExperimentError(f"{context}: fen_after does not match the game result")

    final_board = getattr(game_result, "board", None)
    if not isinstance(final_board, chess.Board):
        raise ExperimentError("game_result.board must be a python-chess Board")
    if final_board.fen() != board.fen():
        raise ExperimentError("game_result.board does not match its move records")
    stopped_at_limit = getattr(game_result, "stopped_at_limit", False)
    records.append(
        _make_result_record(
            board,
            len(records) - 1,
            seq=len(records),
            stopped_at_limit=stopped_at_limit,
            finished_at=finished_at or _timestamp(timestamp_factory),
        )
    )
    # Validate before the caller can persist anything.  This also guarantees
    # that a future change to the event construction cannot emit bad JSONL.
    _validate_records(records)
    return records


def _atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Atomically replace *path* with *text* using a same-directory temp file."""

    target = Path(path)
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ExperimentError(f"parent directory does not exist: {parent}")
    if target.exists() and target.is_dir():
        raise ExperimentError(f"cannot atomically write a directory: {target}")

    temporary_name: str | None = None
    file_descriptor: int | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(parent)
        )
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            file_descriptor = None
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None

        # Persist the directory entry as well where the platform supports
        # directory fsync.  The rename itself remains atomic if fsync is not
        # available (for example, on some non-POSIX filesystems).
        try:
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return target
    except OSError as exc:
        raise ExperimentError(f"unable to atomically write {target}: {exc}") from exc
    finally:
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError:
                pass
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Atomically write one strict JSON object followed by a newline.

    The temporary file is created beside the destination, flushed and fsynced,
    then installed with ``os.replace``.  A failed serialization or write does
    not truncate an existing destination.
    """

    if not isinstance(payload, Mapping):
        raise ExperimentError("JSON result payload must be a mapping")
    _validate_json_value(payload, "JSON result")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentError(f"JSON result is not serializable: {exc}") from exc
    return _atomic_write_text(path, encoded + "\n")


def write_json_result(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Write a single JSON result atomically; alias-friendly public name."""

    return atomic_write_json(path, payload)


def write_result(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Short alias for :func:`write_json_result`."""

    return write_json_result(path, payload)


def _encode_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        _validate_json_value(record, "experiment event")
        lines.append(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return "\n".join(lines) + "\n"


def write_experiment(
    path: str | os.PathLike[str],
    game_result: Any,
    *,
    seed: int,
    engine_settings: Mapping[str, Any],
    experiment_id: str | None = None,
    created_at: str | None = None,
    finished_at: str | None = None,
    timestamp_factory: Callable[[], Any] | None = None,
    result_path: str | os.PathLike[str] | None = None,
) -> ReplayedExperiment:
    """Persist a complete :class:`flychess.game.GameResult` as JSONL.

    The log is validated in memory before the atomic replacement.  If
    ``result_path`` is supplied, the final result record is also written as a
    separate single JSON object with :func:`atomic_write_json`.
    """

    records = _records_from_game_result(
        game_result,
        seed=seed,
        engine_settings=engine_settings,
        experiment_id=experiment_id,
        created_at=created_at,
        finished_at=finished_at,
        timestamp_factory=timestamp_factory,
    )
    _atomic_write_text(path, _encode_jsonl(records))
    if result_path is not None:
        if Path(path).absolute() == Path(result_path).absolute():
            raise ExperimentError("result_path must be different from the JSONL log path")
        atomic_write_json(result_path, records[-1])
    return _validate_records(records)


def record_game(
    path: str | os.PathLike[str],
    game_result: Any,
    **kwargs: Any,
) -> ReplayedExperiment:
    """Record a completed game; convenience wrapper for :func:`write_experiment`."""

    return write_experiment(path, game_result, **kwargs)


class ExperimentRecorder:
    """Incremental recorder for callers that do not use ``play_game``.

    Each mutation rewrites the current records through an atomic same-
    directory replacement.  The file becomes replayable once ``finish`` is
    called; before then it intentionally contains no result record and is
    rejected as incomplete by :func:`read_experiment`.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        start_fen: str = chess.Board().fen(),
        fly_color: chess.Color = chess.WHITE,
        seed: int,
        engine_settings: Mapping[str, Any],
        max_plies: int = 160,
        experiment_id: str | None = None,
        created_at: str | None = None,
        timestamp_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.path = Path(path)
        if not _is_int(seed):
            raise ExperimentError("seed must be an integer")
        _require_sequence(max_plies, "max_plies", "recorder", minimum=1)
        board = _board_from_fen(start_fen, "start_fen", "recorder")
        color_name = _color_name(fly_color)
        settings = _copy_json_mapping(engine_settings, "engine_settings", "recorder")
        self._timestamp_factory = timestamp_factory
        self._metadata = {
            "type": METADATA,
            "schema": SCHEMA,
            "seq": 0,
            "experiment_id": experiment_id or uuid.uuid4().hex,
            "created_at": created_at or _timestamp(timestamp_factory),
            "start_fen": board.fen(),
            "fly_color": color_name,
            "seed": seed,
            "engine_settings": settings,
            "max_plies": max_plies,
        }
        self._board = board
        self._records: list[dict[str, Any]] = [self._metadata]
        self._finished = False
        self._persist_partial()

    @property
    def board(self) -> chess.Board:
        """Return a copy of the current board."""

        return self._board.copy(stack=True)

    @property
    def moves(self) -> tuple[Mapping[str, Any], ...]:
        """Return the recorded move events so far."""

        return tuple(self._records[1:])

    def _ensure_open(self) -> None:
        if self._finished:
            raise ExperimentError("experiment recorder is already finished")

    def _persist_partial(self) -> None:
        _atomic_write_text(self.path, _encode_jsonl(self._records))

    def record_move(
        self,
        move: chess.Move | str,
        *,
        actor: str | None = None,
        timestamp: str | None = None,
    ) -> Mapping[str, Any]:
        """Record one legal move and atomically persist the incomplete log."""

        self._ensure_open()
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError as exc:
                raise ExperimentError(f"invalid UCI move {move!r}") from exc
        if not isinstance(move, chess.Move):
            raise ExperimentError("move must be a chess.Move or UCI string")
        expected_actor = _expected_actor(self._board, self._metadata["fly_color"])
        if actor is None:
            actor = expected_actor
        if actor != expected_actor:
            raise ExperimentError(f"actor {actor!r} is invalid; expected {expected_actor!r}")
        if move not in self._board.legal_moves:
            raise IllegalMoveError(f"illegal move {move.uci()!r} for current recorder board")
        fen_before = self._board.fen()
        san = self._board.san(move)
        self._board.push(move)
        event = {
            "type": MOVE,
            "seq": len(self._records),
            "ply": len(self._records),
            "timestamp": timestamp or _timestamp(self._timestamp_factory),
            "actor": actor,
            "fen_before": fen_before,
            "uci": move.uci(),
            "san": san,
            "fen_after": self._board.fen(),
        }
        if event["ply"] > self._metadata["max_plies"]:
            self._board.pop()
            raise ExperimentError("move exceeds recorder max_plies")
        self._records.append(event)
        self._persist_partial()
        return dict(event)

    def finish(
        self,
        *,
        stopped_at_limit: bool = False,
        finished_at: str | None = None,
        result_path: str | os.PathLike[str] | None = None,
    ) -> ReplayedExperiment:
        """Append the final result, persist, and validate the complete log."""

        self._ensure_open()
        result = _make_result_record(
            self._board,
            len(self._records) - 1,
            seq=len(self._records),
            stopped_at_limit=stopped_at_limit,
            finished_at=finished_at or _timestamp(self._timestamp_factory),
        )
        self._records.append(result)
        self._finished = True
        _validate_records(self._records)
        _atomic_write_text(self.path, _encode_jsonl(self._records))
        if result_path is not None:
            atomic_write_json(result_path, result)
        return _validate_records(self._records)


# A name that reads naturally in applications that prefer ``save_*`` verbs.
save_experiment = write_experiment
