"""Candidate-isolated native Cordis workspace for the H1 proposer."""
from __future__ import annotations

import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from outer.compiling.materialize import materialize
from outer.genome import harness_spec as hs
from outer.safety.static_gates import check_cordis_plugin


_MAX_READ_CHARS = 40_000
_MAX_WRITE_CHARS = 40_000
_PROVENANCE_FILES = {"spec.yaml", "meta.json", "component_manifest.json"}


@dataclass
class WorkspaceCheck:
    valid: bool
    errors: List[str] = field(default_factory=list)
    partial: Optional[Dict[str, Any]] = None
    effective: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = field(default_factory=list)
    component_audit: List[Dict[str, Any]] = field(default_factory=list)


def _resolve(root: Path, raw_path: str, *, must_exist: bool = False) -> Path:
    root = Path(root).resolve()
    raw = str(raw_path or ".").strip()
    path = Path(raw)
    if path.is_absolute():
        raise ValueError("absolute paths are not allowed; use paths relative to H2")
    candidate = (root / path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes the candidate H2 workspace") from exc
    if must_exist and not candidate.exists():
        raise ValueError(f"no such H2 path: {raw}")
    return candidate


def _rel(root: Path, path: Path) -> str:
    value = path.relative_to(Path(root).resolve()).as_posix()
    return value or "."


def relative_path(root: Path, raw_path: str, *, must_exist: bool = False) -> str:
    return _rel(root, _resolve(root, raw_path, must_exist=must_exist))


def _mutable_path(relative: str) -> bool:
    if relative == "cordis.yml":
        return True
    path = Path(relative)
    return (
        len(path.parts) == 2
        and path.parts[0] == "plugins"
        and path.suffix == ".mjs"
        and path.name != "sah-bridge.mjs"
    )


def inspect(root: Path, command: str) -> str:
    """Run one read-only shell-shaped command: pwd, ls, cat, find, or tree."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"ERROR: cannot parse command: {exc}"
    if not argv:
        return "ERROR: empty command"
    op = argv[0]
    try:
        if op == "pwd":
            if len(argv) != 1:
                raise ValueError("usage: pwd")
            return "."
        if op == "ls":
            positional = [arg for arg in argv[1:] if not arg.startswith("-")]
            if len(positional) > 1:
                raise ValueError("usage: ls [-la] [relative-directory]")
            target = _resolve(root, positional[0] if positional else ".", must_exist=True)
            if not target.is_dir():
                return _rel(root, target)
            long_form = any("l" in arg for arg in argv[1:] if arg.startswith("-"))
            rows = []
            for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name)):
                name = child.name + ("/" if child.is_dir() else "")
                rows.append(
                    f"{('-' if child.is_dir() else child.stat().st_size):>8}  {name}"
                    if long_form else name
                )
            return "\n".join(rows) or "(empty directory)"
        if op == "cat":
            if len(argv) != 2:
                raise ValueError("usage: cat relative-file")
            target = _resolve(root, argv[1], must_exist=True)
            if not target.is_file():
                raise ValueError(f"not a file: {argv[1]}")
            text = target.read_text(encoding="utf-8", errors="replace")
            return text if len(text) <= _MAX_READ_CHARS \
                else text[:_MAX_READ_CHARS] + "\n...[read cap reached]"
        if op in {"find", "tree"}:
            if len(argv) > 2:
                raise ValueError(f"usage: {op} [relative-directory]")
            target = _resolve(root, argv[1] if len(argv) == 2 else ".", must_exist=True)
            if not target.is_dir():
                return _rel(root, target)
            values = []
            for path in sorted(target.rglob("*")):
                rel = _rel(root, path)
                values.append(rel + ("/" if path.is_dir() else ""))
            return "\n".join(values) or "(empty directory)"
        raise ValueError("allowed commands: pwd, ls, cat, find, tree")
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


def _write_target(root: Path, raw_path: str) -> tuple[Path, str]:
    target = _resolve(root, raw_path)
    relative = _rel(root, target)
    if not _mutable_path(relative):
        raise ValueError(
            "mutable Cordis surface is only cordis.yml and plugins/*.mjs"
        )
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise ValueError("mutable target must be a regular file")
    return target, relative


def write_file(root: Path, path: str, content: str) -> str:
    try:
        target, relative = _write_target(root, path)
        if not isinstance(content, str):
            raise ValueError("content must be text")
        if len(content) > _MAX_WRITE_CHARS:
            raise ValueError(f"file exceeds {_MAX_WRITE_CHARS}-character write cap")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n", encoding="utf-8")
        return f"WROTE {relative} ({target.stat().st_size} bytes)"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


def edit_file(root: Path, path: str, old_text: str, new_text: str,
              append_text: str) -> str:
    try:
        target, relative = _write_target(root, path)
        if not target.is_file():
            raise ValueError(f"no such mutable file: {relative}")
        current = target.read_text(encoding="utf-8")
        replace_mode = bool(old_text)
        append_mode = bool(append_text)
        if replace_mode == append_mode:
            raise ValueError("provide exactly one of old_text or append_text")
        if replace_mode:
            count = current.count(old_text)
            if count != 1:
                raise ValueError(f"old_text must occur exactly once (found {count})")
            updated = current.replace(old_text, new_text, 1)
        else:
            updated = current.rstrip() + "\n" + append_text.rstrip() + "\n"
        if len(updated) > _MAX_WRITE_CHARS:
            raise ValueError(f"edit exceeds {_MAX_WRITE_CHARS}-character write cap")
        target.write_text(updated, encoding="utf-8")
        return f"EDITED {relative} ({len(current)} -> {len(updated)} chars)"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


def delete_file(root: Path, path: str) -> str:
    try:
        target, relative = _write_target(root, path)
        if relative == "cordis.yml":
            raise ValueError("cordis.yml is required and cannot be deleted")
        if not target.is_file():
            raise ValueError(f"no such mutable file: {relative}")
        target.unlink()
        return f"DELETED {relative}"
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"


def _component_audit(effective: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for field, kind in (
        ("new_tools", "tool"), ("new_skills", "skill"),
        ("new_middlewares", "middleware"),
    ):
        for component in effective.get(field) or []:
            rows.append({
                "kind": kind,
                "name": component["name"],
                "status": "mounted-and-gated",
                "runtime": "cordis-plugin",
            })
    return rows


def _validate_tree(root: Path, errors: List[str]) -> None:
    allowed_top = {"cordis.yml", "plugins", *_PROVENANCE_FILES}
    for entry in root.iterdir():
        if entry.name not in allowed_top:
            errors.append(f"unexpected H2 path outside Cordis surface: {entry.name}")
        if entry.is_symlink():
            errors.append(f"symlinks are forbidden in H2 packages: {entry.name}")
    plugins = root / "plugins"
    if not plugins.is_dir() or plugins.is_symlink():
        errors.append("missing regular plugins/ directory")
        return
    for entry in plugins.iterdir():
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".mjs":
            errors.append(f"plugins/ may contain only regular .mjs files: {entry.name}")
        if entry.name == "sah-bridge.mjs":
            errors.append("candidate may not shadow trusted plugins/sah-bridge.mjs")


def validate_workspace(root: Path, base_spec: Dict[str, Any]) -> WorkspaceCheck:
    root = Path(root).resolve()
    errors: List[str] = []
    if not (root / "cordis.yml").is_file():
        return WorkspaceCheck(False, ["missing required H2 file: cordis.yml"])
    _validate_tree(root, errors)
    try:
        raw_effective = hs.read_base_spec(root, verify_provenance=False)
    except Exception as exc:
        return WorkspaceCheck(False, errors + [f"cannot parse Cordis H2 workspace: {exc}"])

    audit = _component_audit(raw_effective)
    for row in raw_effective.get("new_tools") or []:
        ok, gate_errors = check_cordis_plugin(
            row["implementation_js"], kind="tool", name=row["name"],
        )
        if not ok:
            errors.append(
                f"generated tool {row['name']!r} failed Cordis gate: "
                + "; ".join(gate_errors)
            )
    for row in raw_effective.get("new_middlewares") or []:
        ok, gate_errors = check_cordis_plugin(
            row["implementation_js"], kind="middleware", name=row["name"],
            hook=row.get("hook"),
        )
        if not ok:
            errors.append(
                f"generated middleware {row['name']!r} failed Cordis gate: "
                + "; ".join(gate_errors)
            )
    errors.extend(hs.component_prompt_issues(raw_effective))

    partial = hs.diff_to_partial(raw_effective, base_spec)
    differs, changed = hs.differs_from_base(raw_effective, base_spec)
    if not differs:
        errors.append("workspace is identical to the current Cordis H2 (no-op)")
    validation = hs.parse_and_validate(
        yaml.safe_dump(partial, sort_keys=False, allow_unicode=True, width=100)
    )
    if not validation.valid:
        errors.extend(validation.errors)
        effective = raw_effective
    else:
        effective = hs.merge_with_base(validation.spec or {}, base_spec)
        if hs.canonical_json(effective) != hs.canonical_json(raw_effective):
            errors.append("workspace edits cannot be represented by the Cordis H2 genome")

    # Compile the semantic genome and compare the complete live composition.
    # This catches unknown Cordis rows/config, missing mounts, orphan plugins,
    # stale skill metadata, and alternate filenames in one deterministic check.
    with tempfile.TemporaryDirectory(prefix="h2_cordis_canonical_") as td:
        canonical = Path(td) / "h2"
        try:
            materialize(effective, canonical, meta={"effective": effective})
        except Exception as exc:
            errors.append(f"canonical Cordis H2 compilation failed: {exc}")
        else:
            try:
                actual_doc = yaml.safe_load((root / "cordis.yml").read_text(encoding="utf-8"))
                canonical_doc = yaml.safe_load((canonical / "cordis.yml").read_text(encoding="utf-8"))
                if actual_doc != canonical_doc:
                    errors.append(
                        "cordis.yml is not the canonical composition for its declared genome"
                    )
            except Exception as exc:
                errors.append(f"cannot compare canonical cordis.yml: {exc}")
            actual_plugins = {
                path.name: path.read_text(encoding="utf-8")
                for path in (root / "plugins").glob("*.mjs") if path.is_file()
            }
            canonical_plugins = {
                path.name: path.read_text(encoding="utf-8")
                for path in (canonical / "plugins").glob("*.mjs")
            }
            if set(actual_plugins) != set(canonical_plugins):
                missing = sorted(set(canonical_plugins) - set(actual_plugins))
                orphan = sorted(set(actual_plugins) - set(canonical_plugins))
                if missing:
                    errors.append(f"mounted plugin files missing: {missing}")
                if orphan:
                    errors.append(f"unmounted/orphan plugin files: {orphan}")
            for name in sorted(set(actual_plugins) & set(canonical_plugins)):
                if actual_plugins[name].rstrip() != canonical_plugins[name].rstrip():
                    errors.append(f"plugin bytes disagree with declared genome: plugins/{name}")

    return WorkspaceCheck(
        not errors,
        errors=errors,
        partial=validation.spec if validation.valid else partial,
        effective=effective,
        changed_fields=changed,
        component_audit=audit,
    )
