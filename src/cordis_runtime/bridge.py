"""Authenticated loopback bridge from Cordis tools to SAH's Python sessions.

Cordis owns the model loop and plugin lifecycle.  The evaluator, edit ledger,
workspace protocol, and reward bookkeeping remain Python state.  Each rollout
therefore gets a short-lived HTTP server bound to ``127.0.0.1`` and protected
by a random bearer token.  Calls are serialized because both InnerSession and
ProposeSession are intentionally single-agent state machines.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, ContextManager, Mapping, Optional
from urllib.parse import unquote


ToolCallable = Callable[..., Any]
ScopeFactory = Callable[[], ContextManager[Any]]


@dataclass(frozen=True)
class BridgeCall:
    """One settled bridge dispatch, retained for provenance and tests."""

    sequence: int
    request_id: str
    tool: str
    arguments: dict[str, Any]
    ok: bool
    elapsed_ms: float
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "request_id": self.request_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "ok": self.ok,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "error": self.error,
        }


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], bridge: "BridgeServer") -> None:
        self.bridge = bridge
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _reply(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.bridge.token}"
        return secrets.compare_digest(self.headers.get("Authorization", ""), expected)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path != "/health":
            self._reply(404, {"ok": False, "error": "not found"})
            return
        if not self._authorized():
            self._reply(401, {"ok": False, "error": "unauthorized"})
            return
        self._reply(200, {"ok": True, "role": self.server.bridge.role})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if not self._authorized():
            self._reply(401, {"ok": False, "error": "unauthorized"})
            return
        prefix = "/v1/tools/"
        if not self.path.startswith(prefix) or "/" in self.path[len(prefix):]:
            self._reply(404, {"ok": False, "error": "not found"})
            return
        tool = unquote(self.path[len(prefix):])
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.server.bridge.max_request_bytes:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be a JSON object")
            request_id = payload.get("request_id", "")
            if not isinstance(request_id, str) or len(request_id) > 256:
                raise ValueError("request_id must be a short string")
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            self._reply(400, {"ok": False, "error": str(exc)})
            return

        try:
            value = self.server.bridge.dispatch(tool, arguments, request_id)
        except KeyError:
            self._reply(404, {"ok": False, "error": f"unknown tool: {tool}"})
        except Exception as exc:  # the JS tool pipeline records this as an error
            self._reply(
                500,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )
        else:
            self._reply(200, {"ok": True, "result": value})


class BridgeServer:
    """Context-managed authenticated server for one Cordis rollout."""

    def __init__(
        self,
        *,
        role: str,
        tools: Mapping[str, ToolCallable],
        scope_factory: Optional[ScopeFactory] = None,
        max_request_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if role not in {"inner", "outer"}:
            raise ValueError(f"unsupported Cordis bridge role: {role!r}")
        if not tools or any(not name or "/" in name for name in tools):
            raise ValueError("bridge tools need non-empty, path-safe names")
        if max_request_bytes < 1024:
            raise ValueError("max_request_bytes is too small")
        self.role = role
        self.tools = dict(tools)
        self.scope_factory = scope_factory or nullcontext
        self.max_request_bytes = max_request_bytes
        self.token = secrets.token_urlsafe(32)
        self._dispatch_lock = threading.RLock()
        self._audit_lock = threading.Lock()
        self._audit: list[BridgeCall] = []
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("bridge server is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def audit(self) -> list[dict[str, Any]]:
        with self._audit_lock:
            return [row.as_dict() for row in self._audit]

    def __enter__(self) -> "BridgeServer":
        if self._server is not None:
            raise RuntimeError("bridge server is already running")
        self._server = _Server(("127.0.0.1", 0), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"sah-cordis-{self.role}-bridge",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def dispatch(
        self, tool: str, arguments: dict[str, Any], request_id: str = ""
    ) -> Any:
        """Invoke a registered tool inside its Python session scope."""

        function = self.tools.get(tool)
        if function is None:
            raise KeyError(tool)
        started = time.monotonic()
        ok = False
        error: Optional[str] = None
        try:
            with self._dispatch_lock:
                with self.scope_factory():
                    result = function(**arguments)
            # Validate at the trust boundary rather than letting the response
            # serializer fail after the call has already mutated session state.
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            ok = True
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = (time.monotonic() - started) * 1000
            with self._audit_lock:
                self._audit.append(
                    BridgeCall(
                        sequence=len(self._audit),
                        request_id=request_id,
                        tool=tool,
                        arguments=dict(arguments),
                        ok=ok,
                        elapsed_ms=elapsed,
                        error=error,
                    )
                )
