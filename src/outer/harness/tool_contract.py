"""Canonical model-visible H1 tool schemas for replay/training provenance."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


_EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}

H1_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "harness_shell",
            "description": (
                "Inspect the private candidate Cordis harness with one read-only "
                "command: pwd, ls, cat, find, or tree. Start with cat cordis.yml."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_harness_file",
            "description": (
                "Create or completely replace cordis.yml or one plugins/*.mjs "
                "file. Read existing files before overwriting them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_harness_file",
            "description": (
                "Modify one inspected Cordis file by one exact replacement or "
                "append a small section."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "append_text": {"type": "string"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_harness_file",
            "description": (
                "Delete one inspected plugins/*.mjs file after removing its "
                "cordis.yml mount. Core files cannot be deleted."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_harness",
            "description": "Parse, gate, and canonically compile the Cordis H2 workspace.",
            "parameters": _EMPTY,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_harness",
            "description": "Submit the successfully validated Cordis H2 and end H1.",
            "parameters": _EMPTY,
        },
    },
]


def h1_tool_schemas() -> list[dict[str, Any]]:
    return deepcopy(H1_TOOL_SCHEMAS)
