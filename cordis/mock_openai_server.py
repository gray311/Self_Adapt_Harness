#!/usr/bin/env python3
"""Small deterministic OpenAI chat-completions server for Cordis smoke tests."""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    model = "cordis-smoke-model"
    response_text = "CORDIS_MODEL_OK"
    tool_name: str | None = None
    tool_arguments = "{}"
    request_file: Path | None = None
    request_lock = Lock()
    response_script: list[Dict[str, Any]] = []
    request_count = 0

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") != "/v1/models":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        self._send_json(200, {
            "object": "list",
            "data": [{"id": self.model, "object": "model", "owned_by": "sah"}],
        })

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8 * 1024 * 1024:
                raise ValueError("invalid request length")
            request = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {"message": str(exc)}})
            return

        with self.request_lock:
            request_index = self.request_count
            type(self).request_count += 1
            if self.request_file is not None:
                with self.request_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(request, sort_keys=True) + "\n")

        scripted = (
            self.response_script[min(request_index, len(self.response_script) - 1)]
            if self.response_script else {}
        )
        tool_name = scripted.get("tool_name", self.tool_name)
        tool_arguments = scripted.get("tool_arguments", self.tool_arguments)
        response_text = scripted.get("response", self.response_text)
        if isinstance(tool_arguments, dict):
            tool_arguments = json.dumps(tool_arguments, separators=(",", ":"))

        if request.get("stream") is not True:
            self._send_json(400, {"error": {"message": "stream=true is required"}})
            return

        now = int(time.time())
        common = {
            "id": "chatcmpl-cordis-smoke",
            "object": "chat.completion.chunk",
            "created": now,
            "model": request.get("model", self.model),
        }
        if tool_name:
            first_delta = {
                "role": "assistant",
                "tool_calls": [{
                    "index": 0,
                    "id": f"call-cordis-smoke-{request_index}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_arguments,
                    },
                }],
            }
            finish_reason = "tool_calls"
        else:
            first_delta = {"role": "assistant", "content": response_text}
            finish_reason = "stop"
        chunks = [
            {**common, "choices": [{
                "index": 0,
                "delta": first_delta,
                "finish_reason": None,
            }]},
            {**common, "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }]},
            {**common, "choices": [], "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 3,
                "total_tokens": 11,
            }},
        ]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(b"data: " + _json_bytes(chunk) + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="cordis-smoke-model")
    parser.add_argument("--response", default="CORDIS_MODEL_OK")
    parser.add_argument("--tool-name")
    parser.add_argument("--tool-arguments", default="{}")
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--script-file", type=Path)
    args = parser.parse_args()

    Handler.model = args.model
    Handler.response_text = args.response
    Handler.tool_name = args.tool_name
    Handler.tool_arguments = args.tool_arguments
    Handler.request_file = args.request_file
    if args.script_file:
        script = json.loads(args.script_file.read_text(encoding="utf-8"))
        if not isinstance(script, list) or not all(isinstance(row, dict) for row in script):
            raise SystemExit("--script-file must contain a JSON list of response objects")
        Handler.response_script = script
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    args.ready_file.write_text(str(server.server_port), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
