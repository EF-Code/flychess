"""Local web interface for bounded fly-vs-Stockfish experiments.

The HTTP layer deliberately stays small and dependency-free. A browser can
request the current state, start one bounded game in the background, and poll
for live policy telemetry. The legacy synchronous game endpoint remains
available for callers that need one complete response. The engine executable
is server configuration, never request data, and the only request options
accepted are validated, bounded experiment settings.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import chess

from .brain import SurrogateFlyBrain
from .connectome import load_connectome
from .connectome_policy import ConnectomeFlyBrain
from .engine import StockfishEngine
from .game import DecisionRecord, GameResult, MoveRecord, ProgressCallback, play_game


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_DEPTH = 2
DEFAULT_MAX_PLIES = 24
MIN_DEPTH = 1
MAX_DEPTH = 6
MIN_MAX_PLIES = 1
MAX_MAX_PLIES = 80
MIN_SEED = -1_000_000
MAX_SEED = 1_000_000
MAX_REQUEST_BYTES = 16 * 1024
MAX_RECENT_MOVES = 24
MAX_DECISION_TRACE = 12
MAX_CANDIDATES = 5

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKOUT_WEB_ROOT = _PROJECT_ROOT / "web"
_INSTALLED_WEB_ROOT = Path(sys.prefix) / "share" / "flychess" / "web"
WEB_ROOT = _CHECKOUT_WEB_ROOT if _CHECKOUT_WEB_ROOT.is_dir() else _INSTALLED_WEB_ROOT
_ASSETS: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class WebRequestError(ValueError):
    """A client error with an HTTP status suitable for a JSON response."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


class BusyGameError(RuntimeError):
    """Raised when a second game is requested while one is already running."""


class GameExecutionError(RuntimeError):
    """Raised after a game callback fails and the app records an error state."""


class GameOptions:
    """Validated options accepted by the web endpoint."""

    __slots__ = ("depth", "max_plies", "fly_color", "seed", "policy", "neural_steps")

    def __init__(
        self,
        depth: int,
        max_plies: int,
        fly_color: str,
        seed: int,
        *,
        policy: str = "surrogate",
        neural_steps: int = 2,
    ) -> None:
        self.depth = depth
        self.max_plies = max_plies
        self.fly_color = fly_color
        self.seed = seed
        self.policy = policy
        self.neural_steps = neural_steps

    def __repr__(self) -> str:
        return (
            "GameOptions("
            f"depth={self.depth}, max_plies={self.max_plies}, "
            f"fly_color={self.fly_color!r}, seed={self.seed}, policy={self.policy!r}, "
            f"neural_steps={self.neural_steps})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "max_plies": self.max_plies,
            "fly_color": self.fly_color,
            "seed": self.seed,
            "policy": self.policy,
            "neural_steps": self.neural_steps,
        }


GameCallback = Callable[[str, GameOptions], GameResult]
GameRunner = Callable[[str, GameOptions, ProgressCallback | None], GameResult]


def _bounded_int(value: Any, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WebRequestError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise WebRequestError(f"{name} must be between {minimum} and {maximum}")
    return value


def _validate_engine_path(path: str) -> str:
    """Validate server configuration without ever treating it as shell text."""

    if not isinstance(path, str) or not path.strip():
        raise ValueError("engine_path must be a non-empty string")
    if "\x00" in path or "\r" in path or "\n" in path:
        raise ValueError("engine_path contains an invalid control character")
    return path


def validate_options(
    *,
    depth: Any = DEFAULT_DEPTH,
    max_plies: Any = DEFAULT_MAX_PLIES,
    fly_color: Any = "white",
    seed: Any = 17,
    policy: str = "surrogate",
    neural_steps: Any = 2,
) -> GameOptions:
    """Validate options used by both the server constructor and HTTP input."""

    validated_depth = _bounded_int(depth, name="depth", minimum=MIN_DEPTH, maximum=MAX_DEPTH)
    validated_max_plies = _bounded_int(
        max_plies,
        name="max_plies",
        minimum=MIN_MAX_PLIES,
        maximum=MAX_MAX_PLIES,
    )
    if not isinstance(fly_color, str) or fly_color not in {"white", "black"}:
        raise WebRequestError("fly_color must be 'white' or 'black'")
    validated_seed = _bounded_int(seed, name="seed", minimum=MIN_SEED, maximum=MAX_SEED)
    if policy not in {"surrogate", "connectome"}:
        raise ValueError("policy must be 'surrogate' or 'connectome'")
    validated_neural_steps = _bounded_int(neural_steps, name="neural_steps", minimum=1, maximum=100)
    return GameOptions(
        validated_depth,
        validated_max_plies,
        fly_color,
        validated_seed,
        policy=policy,
        neural_steps=validated_neural_steps,
    )


def parse_game_options(payload: Any, defaults: GameOptions) -> GameOptions:
    """Parse the small, intentionally closed request schema."""

    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise WebRequestError("request body must be a JSON object")

    allowed = {"depth", "max_plies", "fly_color", "seed"}
    for key in payload:
        if key not in allowed:
            raise WebRequestError(f"unsupported option: {key}")

    return validate_options(
        depth=payload.get("depth", defaults.depth),
        max_plies=payload.get("max_plies", defaults.max_plies),
        fly_color=payload.get("fly_color", defaults.fly_color),
        seed=payload.get("seed", defaults.seed),
        policy=defaults.policy,
        neural_steps=defaults.neural_steps,
    )


def _validate_connectome_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("connectome_path must be a non-empty path")
    try:
        candidate = Path(path).expanduser().resolve()
    except (OSError, ValueError) as exc:
        raise ValueError("connectome_path is invalid") from exc
    if not candidate.is_file():
        raise ValueError(f"connectome file does not exist: {candidate}")
    try:
        load_connectome(candidate)
    except Exception as exc:
        raise ValueError(f"connectome file is invalid: {exc}") from exc
    return str(candidate)


def run_bounded_game(
    engine_path: str,
    options: GameOptions,
    connectome_path: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> GameResult:
    """Run one server-authorized game using the existing package APIs.

    ``engine_path`` comes from the process that started the web server.  It is
    intentionally not exposed in the JSON request schema.  ``StockfishEngine``
    owns the UCI process lifecycle and does not invoke a shell.
    """

    fly_color = chess.WHITE if options.fly_color == "white" else chess.BLACK
    if connectome_path is None:
        fly = SurrogateFlyBrain(seed=options.seed)
    else:
        fly = ConnectomeFlyBrain(
            load_connectome(connectome_path),
            steps_per_position=options.neural_steps,
        )
    with StockfishEngine(path=engine_path, depth=options.depth) as stockfish:
        return play_game(
            fly,
            stockfish,
            fly_color=fly_color,
            max_plies=options.max_plies,
            on_progress=on_progress,
        )


def _move_payload(record: MoveRecord) -> dict[str, Any]:
    return {
        "ply": record.ply,
        "actor": record.actor,
        "san": record.san,
        "uci": record.uci,
    }


def _decision_payload(record: DecisionRecord) -> dict[str, Any]:
    readout = record.readout
    return {
        "ply": record.ply,
        "fen": record.fen,
        "policy": readout.policy,
        "selected_uci": readout.selected_uci,
        "candidates": [
            {
                "uci": candidate.uci,
                "san": candidate.san,
                "score": round(candidate.score, 6),
            }
            for candidate in readout.candidates[:MAX_CANDIDATES]
        ],
        "activity": {name: round(value, 6) for name, value in readout.activity},
    }


def _base_state(options: GameOptions, *, status: str, message: str) -> dict[str, Any]:
    is_surrogate = options.policy == "surrogate"
    return {
        "ok": True,
        "status": status,
        "phase": {
            "running": "thinking",
            "complete": "complete",
            "error": "error",
        }.get(status, "idle"),
        "message": message,
        "fen": chess.Board().fen(),
        "recent_moves": [],
        "decision_trace": [],
        "move_count": 0,
        "outcome": None,
        "outcome_detail": None,
        "stopped_at_limit": False,
        "fly": {
            "policy": "SurrogateFlyBrain" if is_surrogate else "ConnectomeFlyBrain",
            "is_surrogate": is_surrogate,
            "color": options.fly_color,
            "seed": options.seed,
            "neural_steps": options.neural_steps,
        },
        "stockfish": {"depth": options.depth},
        "bounds": {"max_plies": options.max_plies},
    }


def result_to_state(result: GameResult, options: GameOptions) -> dict[str, Any]:
    """Convert a game result into the stable JSON shape used by the browser."""

    outcome = result.outcome
    if outcome is None and result.stopped_at_limit:
        status = "complete"
        message = f"Stopped after {len(result.moves)} plies at the configured bound."
        outcome_code = "*"
        outcome_detail = "move limit"
    elif outcome is None:
        status = "complete"
        message = f"Run returned after {len(result.moves)} plies without a terminal outcome."
        outcome_code = "*"
        outcome_detail = "not reported"
    else:
        status = "complete"
        outcome_code = outcome.result()
        outcome_detail = outcome.termination.name.replace("_", " ").lower()
        message = f"Game complete: {outcome_code} ({outcome_detail})."

    state = _base_state(options, status=status, message=message)
    state.update(
        {
            "fen": result.board.fen(),
            "recent_moves": [_move_payload(record) for record in result.moves[-MAX_RECENT_MOVES:]],
            "decision_trace": [
                _decision_payload(record) for record in result.decision_trace[-MAX_DECISION_TRACE:]
            ],
            "move_count": len(result.moves),
            "outcome": outcome_code,
            "outcome_detail": outcome_detail,
            "stopped_at_limit": result.stopped_at_limit,
        }
    )
    return state


def progress_to_state(
    result: GameResult,
    options: GameOptions,
    *,
    phase: str,
    message: str,
) -> dict[str, Any]:
    """Convert an in-flight game snapshot into the browser state shape."""

    state = result_to_state(result, options)
    state["status"] = "running"
    state["phase"] = phase
    state["message"] = message
    state["outcome"] = None
    state["outcome_detail"] = None
    state["stopped_at_limit"] = False
    return state


class FlychessWebApp:
    """Thread-safe application state and server-side game callback boundary."""

    def __init__(
        self,
        *,
        engine_path: str = "stockfish",
        depth: int = DEFAULT_DEPTH,
        max_plies: int = DEFAULT_MAX_PLIES,
        seed: int = 17,
        connectome_path: str | Path | None = None,
        neural_steps: int = 2,
        callback: GameCallback | None = None,
    ) -> None:
        self.engine_path = _validate_engine_path(engine_path)
        self.connectome_path = _validate_connectome_path(connectome_path)
        policy = "connectome" if self.connectome_path is not None else "surrogate"
        self.defaults = validate_options(
            depth=depth,
            max_plies=max_plies,
            seed=seed,
            policy=policy,
            neural_steps=neural_steps,
        )
        self._callback = callback or (
            lambda configured_engine_path, options: run_bounded_game(
                configured_engine_path,
                options,
                self.connectome_path,
            )
        )
        if callback is None:
            self._runner: GameRunner = lambda configured_engine_path, options, on_progress: run_bounded_game(
                configured_engine_path,
                options,
                self.connectome_path,
                on_progress=on_progress,
            )
        else:
            self._runner = lambda configured_engine_path, options, _on_progress: callback(
                configured_engine_path,
                options,
            )
        self._state_lock = threading.Lock()
        self._game_lock = threading.Lock()
        self._state = _base_state(
            self.defaults,
            status="ready",
            message="Ready for a bounded fly-vs-Stockfish game.",
        )

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return copy.deepcopy(self._state)

    def start_game(self, payload: Any = None) -> dict[str, Any]:
        """Validate a request, invoke the fixed callback, and publish its result."""

        options = parse_game_options(payload, self.defaults)
        if not self._game_lock.acquire(blocking=False):
            raise BusyGameError("a game is already running")

        try:
            with self._state_lock:
                self._state = _base_state(
                    options,
                    status="running",
                    message="Running the bounded fly-vs-Stockfish game…",
                )
            try:
                result = self._callback(self.engine_path, options)
            except Exception as exc:
                message = f"Game could not be started: {type(exc).__name__}: {exc}"
                with self._state_lock:
                    error_state = _base_state(options, status="error", message=message)
                    error_state["ok"] = False
                    self._state = error_state
                raise GameExecutionError(message) from exc

            with self._state_lock:
                self._state = result_to_state(result, options)
            return self.snapshot()
        finally:
            self._game_lock.release()

    def start_game_async(self, payload: Any = None) -> dict[str, Any]:
        """Start a game in the background and return its initial state."""

        options = parse_game_options(payload, self.defaults)
        if not self._game_lock.acquire(blocking=False):
            raise BusyGameError("a game is already running")

        with self._state_lock:
            self._state = _base_state(
                options,
                status="running",
                message="Fly is watching the board before its first decision…",
            )
        worker = threading.Thread(
            target=self._run_game_worker,
            args=(options,),
            name="flychess-game",
            daemon=True,
        )
        worker.start()
        return self.snapshot()

    def _run_game_worker(self, options: GameOptions) -> None:
        def publish(result: GameResult, phase: str, message: str) -> None:
            with self._state_lock:
                self._state = progress_to_state(
                    result,
                    options,
                    phase=phase,
                    message=message,
                )

        try:
            result = self._runner(self.engine_path, options, publish)
        except Exception as exc:
            message = f"Game could not be started: {type(exc).__name__}: {exc}"
            with self._state_lock:
                error_state = _base_state(options, status="error", message=message)
                error_state["ok"] = False
                self._state = error_state
        else:
            with self._state_lock:
                final_state = result_to_state(result, options)
                final_state["phase"] = "complete"
                self._state = final_state
        finally:
            self._game_lock.release()


class FlychessRequestHandler(BaseHTTPRequestHandler):
    """HTTP adapter for :class:`FlychessWebApp`."""

    protocol_version = "HTTP/1.1"
    server_version = "flychess/0.1"
    sys_version = ""

    @property
    def app(self) -> FlychessWebApp:
        return self.server.app  # type: ignore[attr-defined]

    def _path(self) -> str:
        # Decode only for allowlist comparison; never join a client path to the
        # filesystem.  The asset names below are fixed server-side constants.
        return unquote(urlsplit(self.path).path)

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _serve_asset(self, *, send_body: bool = True) -> None:
        asset = _ASSETS.get(self._path())
        if asset is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return

        filename, content_type = asset
        root = WEB_ROOT.resolve()
        asset_path = (root / filename).resolve()
        if not asset_path.is_relative_to(root):
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            body = asset_path.read_bytes()
        except OSError:
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, "web asset unavailable")
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename == "index.html":
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _read_json(self) -> Any:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return {}
        try:
            length = int(content_length)
        except ValueError as exc:
            raise WebRequestError("Content-Length must be an integer") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise WebRequestError(
                f"request body must be at most {MAX_REQUEST_BYTES} bytes",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise WebRequestError("request body ended unexpectedly")
        if not raw.strip():
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WebRequestError("request body must be valid UTF-8 JSON") from exc

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        path = self._path()
        if path == "/api/state":
            self._send_json(HTTPStatus.OK, self.app.snapshot())
            return
        self._serve_asset()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self._serve_asset(send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        path = self._path()
        if path not in {"/api/game", "/api/game/start"}:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            payload = self._read_json()
            if path == "/api/game/start":
                state = self.app.start_game_async(payload)
                response_status = HTTPStatus.ACCEPTED
            else:
                state = self.app.start_game(payload)
                response_status = HTTPStatus.OK
        except WebRequestError as exc:
            self._send_error_json(exc.status, str(exc))
        except BusyGameError as exc:
            self._send_error_json(HTTPStatus.CONFLICT, str(exc))
        except GameExecutionError as exc:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": str(exc), "state": self.app.snapshot()},
            )
        else:
            self._send_json(response_status, state)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the local console useful without dumping request bodies.
        super().log_message(format, *args)


class FlychessHTTPServer(ThreadingHTTPServer):
    """Threaded localhost-friendly HTTP server carrying app state."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], app: FlychessWebApp) -> None:
        self.app = app
        super().__init__(server_address, FlychessRequestHandler)


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    engine_path: str = "stockfish",
    depth: int = DEFAULT_DEPTH,
    max_plies: int = DEFAULT_MAX_PLIES,
    seed: int = 17,
    connectome_path: str | Path | None = None,
    neural_steps: int = 2,
    callback: GameCallback | None = None,
) -> FlychessHTTPServer:
    """Create a server; by default it binds only to IPv4 loopback."""

    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    app = FlychessWebApp(
        engine_path=engine_path,
        depth=depth,
        max_plies=max_plies,
        seed=seed,
        connectome_path=connectome_path,
        neural_steps=neural_steps,
        callback=callback,
    )
    return FlychessHTTPServer((host, port), app)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m flychess.web",
        description="Serve the local Flychess board and bounded game runner.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--engine", default="stockfish", help="Stockfish executable path")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_MAX_PLIES)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--connectome", type=Path, default=None, help="Use a validated connectome JSON edge list")
    parser.add_argument("--neural-steps", type=int, default=2, help="Connectome steps per fly turn")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    server = create_server(
        args.host,
        args.port,
        engine_path=args.engine,
        depth=args.depth,
        max_plies=args.max_plies,
        seed=args.seed,
        connectome_path=args.connectome,
        neural_steps=args.neural_steps,
    )
    host, port = server.server_address[:2]
    print(f"flychess web UI: http://{host}:{port}")
    print("Press Ctrl-C to stop. The server binds to loopback by default.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
