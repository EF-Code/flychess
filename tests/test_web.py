"""Tests for the dependency-free local web boundary."""

from __future__ import annotations

import http.client
import json
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from pathlib import Path

import chess
import pytest

from flychess.brain import DecisionReadout, MoveCandidate
from flychess.game import DecisionRecord, GameResult, MoveRecord
from flychess.web import (
    DEFAULT_HOST,
    GameOptions,
    create_server,
    parse_game_options,
    result_to_state,
)


def _fake_result(options: GameOptions) -> GameResult:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    fen_before = board.fen()
    san = board.san(move)
    board.push(move)
    return GameResult(
        board=board,
        moves=[
            MoveRecord(
                ply=1,
                actor="fly",
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=board.fen(),
            )
        ],
        start_fen=fen_before,
        fly_color=chess.WHITE if options.fly_color == "white" else chess.BLACK,
        max_plies=options.max_plies,
    )


@contextmanager
def running_server(callback=None, **kwargs: Any) -> Iterator[tuple[Any, list[GameOptions]]]:
    calls: list[GameOptions] = []

    def recording_callback(engine_path: str, options: GameOptions) -> GameResult:
        assert engine_path == "server-owned-stockfish"
        calls.append(options)
        if callback is not None:
            return callback(engine_path, options)
        return _fake_result(options)

    server = create_server(
        port=0,
        engine_path="server-owned-stockfish",
        callback=recording_callback,
        **kwargs,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(server: Any, method: str, path: str, body: Any = None) -> tuple[int, dict[str, Any] | str]:
    host, port = server.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=10)
    encoded = None if body is None else json.dumps(body)
    headers = {"Accept": "application/json"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    connection.close()
    text = raw.decode("utf-8")
    if response.getheader("Content-Type", "").startswith("application/json"):
        return response.status, json.loads(text)
    return response.status, text


def test_options_are_closed_and_bounded() -> None:
    server = create_server(port=0)
    try:
        defaults = parse_game_options({}, server.app.defaults)
    finally:
        server.server_close()
    assert defaults.depth == 2
    assert defaults.max_plies == 24

    with pytest.raises(ValueError, match="unsupported option"):
        parse_game_options({"engine": "stockfish; touch /tmp/pwned"}, defaults)
    with pytest.raises(ValueError, match="between"):
        parse_game_options({"max_plies": 81}, defaults)
    with pytest.raises(ValueError, match="fly_color"):
        parse_game_options({"fly_color": "shell"}, defaults)


def test_server_defaults_to_loopback_and_serves_allowlisted_assets() -> None:
    with running_server() as (server, _calls):
        assert server.server_address[0] == DEFAULT_HOST

        status, html = request(server, "GET", "/")
        assert status == 200
        assert isinstance(html, str)
        assert "FlyNet Lab" in html
        assert "Surrogate" in html

        status, payload = request(server, "GET", "/api/state")
        assert status == 200
        assert payload["status"] == "ready"
        assert payload["fly"]["is_surrogate"] is True
        assert len(payload["fen"].split(" ")) == 6

        status, _payload = request(server, "GET", "/%2e%2e/README.md")
        assert status == 404


def test_post_game_invokes_server_callback_and_returns_trace() -> None:
    with running_server(max_plies=12) as (server, calls):
        status, payload = request(
            server,
            "POST",
            "/api/game",
            {"depth": 1, "max_plies": 4, "fly_color": "white", "seed": 9},
        )

        assert status == 200
        assert len(calls) == 1
        assert calls[0].depth == 1
        assert calls[0].max_plies == 4
        assert calls[0].seed == 9
        assert payload["status"] == "complete"
        assert payload["move_count"] == 1
        assert payload["recent_moves"][0]["uci"] == "e2e4"
        assert payload["fen"] != chess.Board().fen()


def test_web_state_includes_bounded_decision_readout() -> None:
    board = chess.Board()
    readout = DecisionReadout(
        policy="SurrogateFlyBrain",
        selected_uci="e2e4",
        candidates=(
            MoveCandidate(uci="e2e4", san="e4", score=0.5),
            MoveCandidate(uci="d2d4", san="d4", score=0.25),
        ),
        activity=(("active_nodes", 4.0), ("node_count", 32.0)),
    )
    state = result_to_state(
        GameResult(
            board=board,
            max_plies=4,
            decision_trace=[DecisionRecord(ply=1, fen=board.fen(), readout=readout)],
        ),
        GameOptions(depth=2, max_plies=4, fly_color="white", seed=17),
    )

    assert state["decision_trace"][0]["selected_uci"] == "e2e4"
    assert state["decision_trace"][0]["candidates"][0]["san"] == "e4"
    assert state["decision_trace"][0]["activity"]["active_nodes"] == 4.0


def test_invalid_post_does_not_invoke_callback() -> None:
    with running_server() as (server, calls):
        status, payload = request(server, "POST", "/api/game", {"engine": "stockfish"})
        assert status == 400
        assert "unsupported option" in payload["error"]
        assert calls == []


def test_second_game_is_rejected_while_callback_is_running() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_callback(_engine_path: str, options: GameOptions) -> GameResult:
        entered.set()
        assert release.wait(timeout=2)
        return _fake_result(options)

    with running_server(callback=blocking_callback) as (server, _calls):
        first_result: list[tuple[int, dict[str, Any] | str]] = []

        def run_first() -> None:
            first_result.append(request(server, "POST", "/api/game", {}))

        first_thread = threading.Thread(target=run_first)
        first_thread.start()
        assert entered.wait(timeout=2)

        status, payload = request(server, "POST", "/api/game", {})
        assert status == 409
        assert payload["error"] == "a game is already running"

        release.set()
        first_thread.join(timeout=2)
        assert first_result[0][0] == 200


def test_async_game_start_returns_running_state_and_finishes_in_state_endpoint() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_callback(_engine_path: str, options: GameOptions) -> GameResult:
        entered.set()
        assert release.wait(timeout=2)
        return _fake_result(options)

    with running_server(callback=blocking_callback) as (server, _calls):
        status, state = request(server, "POST", "/api/game/start", {})
        assert status == 202
        assert state["status"] == "running"
        assert state["phase"] == "thinking"
        assert entered.wait(timeout=2)

        status, state = request(server, "GET", "/api/state")
        assert status == 200
        assert state["status"] == "running"

        release.set()
        final_state: dict[str, Any] | None = None
        for _ in range(50):
            status, candidate = request(server, "GET", "/api/state")
            assert status == 200
            if candidate["status"] != "running":
                final_state = candidate
                break
            time.sleep(0.02)

        assert final_state is not None
        assert final_state["status"] == "complete"
        assert final_state["phase"] == "complete"


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="Stockfish is not installed")
def test_server_can_run_the_configured_connectome_policy() -> None:
    connectome_path = Path(__file__).resolve().parents[1] / "examples" / "tiny-connectome.json"
    server = create_server(
        port=0,
        engine_path=shutil.which("stockfish"),
        connectome_path=connectome_path,
        neural_steps=2,
        max_plies=2,
        depth=1,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, state = request(server, "GET", "/api/state")
        assert status == 200
        assert state["fly"]["policy"] == "ConnectomeFlyBrain"
        assert state["fly"]["is_surrogate"] is False

        status, state = request(server, "POST", "/api/game", {})
        assert status == 200
        assert state["fly"]["policy"] == "ConnectomeFlyBrain"
        assert state["move_count"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
