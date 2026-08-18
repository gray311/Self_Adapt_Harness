"""Loopback OpenAI request adapter for Cordis-only missing sampling fields."""
from __future__ import annotations

import http.client
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import SplitResult, urlsplit


@dataclass(frozen=True)
class ModelRequest:
    sequence: int
    method: str
    path: str
    status: int
    elapsed_ms: float
    sampling: dict[str, Any]
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "method": self.method,
            "path": self.path,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "sampling": self.sampling,
            "error": self.error,
        }


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], proxy: "ModelProxy") -> None:
        self.proxy = proxy
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def _forward(self) -> None:
        proxy = self.server.proxy
        started = time.monotonic()
        status = 502
        error: Optional[str] = None
        sampling: dict[str, Any] = {}
        connection: Optional[http.client.HTTPConnection] = None
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length < 0 or length > proxy.max_request_bytes:
                raise ValueError("invalid model request length")
            body = self.rfile.read(length) if length else b""
            target_path = proxy.upstream_path(self.path)
            body, sampling = proxy.rewrite(target_path, body)
            upstream = proxy.upstream
            host = upstream.hostname or ""
            port = upstream.port or (443 if upstream.scheme == "https" else 80)
            connection_cls = (
                http.client.HTTPSConnection
                if upstream.scheme == "https" else http.client.HTTPConnection
            )
            connection = connection_cls(host, port, timeout=proxy.timeout_s)
            excluded = {
                "host", "content-length", "connection", "transfer-encoding",
                "accept-encoding",
            }
            headers = {
                key: value for key, value in self.headers.items()
                if key.lower() not in excluded
            }
            headers["Content-Length"] = str(len(body))
            headers["Accept-Encoding"] = "identity"
            connection.request(self.command, target_path, body=body, headers=headers)
            response = connection.getresponse()
            status = int(response.status)
            self.send_response(response.status, response.reason)
            blocked = {
                "content-length", "transfer-encoding", "connection",
                "content-encoding", "keep-alive",
            }
            for key, value in response.getheaders():
                if key.lower() not in blocked:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not self.wfile.closed:
                try:
                    payload = json.dumps({"error": {"message": error}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                except OSError:
                    pass
        finally:
            if connection is not None:
                connection.close()
            self.close_connection = True
            proxy.record(
                method=self.command,
                path=self.path,
                status=status,
                elapsed_ms=(time.monotonic() - started) * 1000,
                sampling=sampling,
                error=error,
            )


class ModelProxy:
    """Stream-preserving proxy that adds SAH sampling extensions to JSON."""

    def __init__(
        self,
        base_url: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        enable_thinking: Optional[bool] = False,
        timeout_s: float = 600.0,
        max_request_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        upstream = urlsplit(base_url.rstrip("/"))
        if upstream.scheme not in {"http", "https"} or not upstream.hostname:
            raise ValueError(f"invalid model base URL: {base_url!r}")
        self.upstream: SplitResult = upstream
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.timeout_s = timeout_s
        self.max_request_bytes = max_request_bytes
        self._server: Optional[_Server] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._audit: list[ModelRequest] = []

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("model proxy is not running")
        path = self.upstream.path.rstrip("/")
        return f"http://127.0.0.1:{self._server.server_port}{path}"

    @property
    def audit(self) -> list[dict[str, Any]]:
        with self._lock:
            return [row.as_dict() for row in self._audit]

    def upstream_path(self, incoming: str) -> str:
        parsed = urlsplit(incoming)
        base = self.upstream.path.rstrip("/")
        path = parsed.path
        if base and path.startswith(base + "/"):
            suffix = path[len(base):]
        elif base and path == base:
            suffix = ""
        else:
            suffix = path
        target = (base + "/" + suffix.lstrip("/")) or "/"
        return target + (("?" + parsed.query) if parsed.query else "")

    def rewrite(self, path: str, body: bytes) -> tuple[bytes, dict[str, Any]]:
        if not body or not any(
            marker in path for marker in ("/chat/completions", "/completions")
        ):
            return body, {}
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return body, {}
        if not isinstance(payload, dict):
            return body, {}
        configured = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
        }
        sampling: dict[str, Any] = {}
        for key, value in configured.items():
            if value is not None:
                payload[key] = value
                sampling[key] = value
        if self.enable_thinking is not None:
            current = payload.get("chat_template_kwargs")
            kwargs = dict(current) if isinstance(current, dict) else {}
            kwargs["enable_thinking"] = self.enable_thinking
            payload["chat_template_kwargs"] = kwargs
            sampling["enable_thinking"] = self.enable_thinking
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8"), sampling

    def record(self, **values: Any) -> None:
        with self._lock:
            self._audit.append(ModelRequest(sequence=len(self._audit), **values))

    def __enter__(self) -> "ModelProxy":
        self._server = _Server(("127.0.0.1", 0), self)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="sah-cordis-model-proxy",
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
