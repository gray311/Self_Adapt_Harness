"""Launch one isolated DSH/Cordis rollout against an SAH model endpoint."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from cordis_runtime.bridge import BridgeServer, ScopeFactory, ToolCallable
from cordis_runtime.model_proxy import ModelProxy
from cordis_runtime.trajectory import (
    completed_turn,
    cordis_events_to_messages,
    count_model_calls,
    find_top_level_session,
    load_session_log,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CORDIS_RUNNER = REPO_ROOT / "cordis" / "run.sh"


def _stage_candidate_plugins(patch: Path, dsh_home: Path) -> None:
    """Snapshot candidate ``plugins/*.mjs`` into the isolated profile tree.

    Cordis CLI overlays patch an existing profile Include; relative plugin
    names consequently resolve from that profile, not from the overlay file.
    Only the canonical flat plugin directory is copied, and the trusted bridge
    name is reserved for ``cordis/run.sh``.
    """

    source_dir = patch.parent / "plugins"
    if not source_dir.exists():
        return
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise ValueError(f"Cordis plugin root must be a real directory: {source_dir}")
    destination = dsh_home / "profiles" / "headless" / "plugins"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.iterdir()):
        if source.name == "sah-bridge.mjs":
            raise ValueError("candidate may not shadow the trusted sah-bridge plugin")
        if source.is_symlink() or not source.is_file() or source.suffix != ".mjs":
            raise ValueError(f"invalid Cordis candidate plugin entry: {source}")
        shutil.copy2(source, destination / source.name)


@dataclass
class CordisRunResult:
    returncode: int
    completed: bool
    stop_reason: str
    stdout: str
    stderr: str
    run_dir: Path
    raw_session_log: Optional[Path]
    events: list[dict[str, Any]] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    bridge_audit: list[dict[str, Any]] = field(default_factory=list)
    model_request_audit: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    error: Optional[str] = None


def run_cordis(
    task: str,
    *,
    role: str,
    patch: Path,
    tools: Mapping[str, ToolCallable],
    scope_factory: ScopeFactory,
    model: str,
    base_url: str,
    api_key: str = "EMPTY",
    max_tokens: int = 8192,
    max_retries: int = 2,
    temperature: Optional[float] = None,
    request_max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    enable_thinking: Optional[bool] = False,
    context_window: int = 131072,
    timeout_s: float = 600.0,
    workspace: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    max_iterations: Optional[int] = None,
    extra_env: Optional[Mapping[str, str]] = None,
) -> CordisRunResult:
    """Run DSH headless with one candidate Cordis patch and Python tool bridge."""

    patch = Path(patch).resolve()
    if not patch.is_file():
        raise FileNotFoundError(f"Cordis patch not found: {patch}")
    if not task.strip():
        raise ValueError("Cordis task must not be empty")
    if max_tokens <= 0 or context_window <= 0 or timeout_s <= 0:
        raise ValueError("Cordis model/runtime limits must be positive")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("Cordis max_retries must be a non-negative integer")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("Cordis top_p must be in (0, 1]")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("Cordis top_k must be a positive integer")
    if not CORDIS_RUNNER.is_file():
        raise FileNotFoundError(f"Cordis launcher not found: {CORDIS_RUNNER}")

    if run_dir is None:
        run_path = Path(tempfile.mkdtemp(prefix=f"sah-cordis-{role}-"))
    else:
        run_path = Path(run_dir).resolve()
        run_path.mkdir(parents=True, exist_ok=True)
    dsh_home = run_path / "dsh-home"
    trajectory_root = run_path / "cordis-sessions"
    dsh_home.mkdir(parents=True, exist_ok=True)
    trajectory_root.mkdir(parents=True, exist_ok=True)
    _stage_candidate_plugins(patch, dsh_home)

    env = os.environ.copy()
    env.update(
        {
            "OPENAI_BASE_URL": base_url.rstrip("/"),
            "OPENAI_API_KEY": api_key,
            "SAH_CORDIS_MODEL": model,
            "SAH_CORDIS_CONTEXT_WINDOW": str(context_window),
            "SAH_CORDIS_MAX_TOKENS": str(max_tokens),
            "SAH_CORDIS_MAX_RETRIES": str(max_retries),
            "SAH_CORDIS_ROLE": role,
            "SAH_CORDIS_WORKSPACE": str(Path(workspace or REPO_ROOT).resolve()),
            "DSH_HOME": str(dsh_home),
            "SAH_CORDIS_TRAJECTORY_ROOT": str(trajectory_root),
            "DSH_TELEMETRY_DISABLED": "1",
            "DSH_PERMISSION_MODE": "read-only",
        }
    )
    if max_iterations is not None:
        env["SAH_CORDIS_MAX_ITERATIONS"] = str(max_iterations)
    if temperature is not None:
        env["SAH_CORDIS_REQUEST_TEMPERATURE"] = str(temperature)
    if request_max_tokens is not None:
        if request_max_tokens <= 0:
            raise ValueError("request_max_tokens must be positive")
        env["SAH_CORDIS_REQUEST_MAX_TOKENS"] = str(request_max_tokens)
    if top_p is not None:
        env["SAH_CORDIS_REQUEST_TOP_P"] = str(top_p)
    if top_k is not None:
        env["SAH_CORDIS_REQUEST_TOP_K"] = str(top_k)
    if seed is not None:
        env["SAH_CORDIS_REQUEST_SEED"] = str(seed)
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})

    stdout = stderr = ""
    returncode = -1
    launch_error: Optional[str] = None
    with ModelProxy(
        base_url,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        enable_thinking=enable_thinking,
        timeout_s=timeout_s,
    ) as model_proxy, BridgeServer(
        role=role, tools=tools, scope_factory=scope_factory
    ) as bridge:
        env["OPENAI_BASE_URL"] = model_proxy.url
        env["SAH_CORDIS_BRIDGE_URL"] = bridge.url
        env["SAH_CORDIS_BRIDGE_TOKEN"] = bridge.token
        try:
            process = subprocess.run(
                [str(CORDIS_RUNNER), "--patch", str(patch), task],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
            returncode = process.returncode
            stdout, stderr = process.stdout, process.stderr
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            launch_error = f"CordisTimeout: exceeded {timeout_s:g}s"
        except OSError as exc:
            launch_error = f"{type(exc).__name__}: {exc}"
        bridge_audit = bridge.audit
        model_request_audit = model_proxy.audit

    (run_path / "stdout.log").write_text(stdout, encoding="utf-8")
    (run_path / "stderr.log").write_text(stderr, encoding="utf-8")
    (run_path / "bridge-audit.json").write_text(
        json.dumps(bridge_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_path / "model-request-audit.json").write_text(
        json.dumps(model_request_audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    raw_log = find_top_level_session(trajectory_root)
    events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    parse_error: Optional[str] = None
    if raw_log is not None:
        try:
            _header, events = load_session_log(raw_log)
            trajectory = cordis_events_to_messages(events)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parse_error = f"CordisTrajectoryError: {exc}"
    else:
        parse_error = "CordisTrajectoryError: no top-level session log"

    completed = returncode == 0 and completed_turn(events)
    errors = [value for value in (launch_error, parse_error) if value]
    if returncode not in {0, -1}:
        errors.insert(0, f"Cordis exited with status {returncode}")
    if returncode == 0 and events and not completed:
        errors.append("Cordis trajectory has no completed turn")
    error = "; ".join(errors) or None
    stop_reason = "completed" if completed and error is None else "harness_error"
    return CordisRunResult(
        returncode=returncode,
        completed=completed,
        stop_reason=stop_reason,
        stdout=stdout,
        stderr=stderr,
        run_dir=run_path,
        raw_session_log=raw_log,
        events=events,
        trajectory=trajectory,
        bridge_audit=bridge_audit,
        model_request_audit=model_request_audit,
        llm_calls=count_model_calls(events),
        error=error,
    )
