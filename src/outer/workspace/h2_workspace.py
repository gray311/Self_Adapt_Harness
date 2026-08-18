"""Candidate-isolated filesystem workspace for the H1 proposer.

The proposer edits a real materialized H2 package.  Read commands intentionally
look shell-like so the model trajectory is inspect -> decide -> edit, while all
path resolution happens in Python and cannot escape the private candidate root.
The workspace is never executed directly: validation parses its H2 semantics
and recompiles a canonical package before a candidate can be rolled out.
"""
from __future__ import annotations

import copy
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from outer.genome import harness_spec as hs
from outer.compiling.materialize import materialize


_MAX_READ_CHARS = 40_000
_MAX_WRITE_CHARS = 40_000
_CORE_TOOL_NAMES = {"edit_solution", "evaluate_solution", "probe_solution", "finish"}
_FIXED_MIDDLEWARE_FILES = {"__init__.py", "budget_reminder.py", "stall_restart.py"}


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
        raise ValueError("absolute paths are not allowed; paths are relative to the H2 root")
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
    """Canonical workspace-relative path used by the session audit."""

    return _rel(root, _resolve(root, raw_path, must_exist=must_exist))


def _mutable_path(relative: str) -> bool:
    """Only H2-genome files are writable; provenance/runtime are read-only."""

    path = Path(relative)
    parts = path.parts
    if relative in {"agent.yaml", "prompt.md"}:
        return True
    if len(parts) == 2 and parts[0] == "tools" and parts[1].endswith(".tool.yaml"):
        return True
    if len(parts) == 2 and parts[0] == "custom_tools" and parts[1].endswith(".py"):
        return True
    if (len(parts) == 3 and parts[0] == "skills" and parts[2] == "SKILL.md"):
        return True
    if len(parts) == 2 and parts[0] == "middlewares":
        name = parts[1]
        # Built-in implementation is fixed runtime.  Its parameters remain
        # proposer-owned through agent.yaml; evolved middleware files are H2.
        return (
            name not in _FIXED_MIDDLEWARE_FILES
            and (name.endswith(".py") or name.endswith(".middleware.yaml"))
        )
    return False


def inspect(root: Path, command: str) -> str:
    """Run one read-only, shell-shaped command: pwd, ls, cat, find, or tree."""

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
            rows = []
            long_form = any("l" in arg for arg in argv[1:] if arg.startswith("-"))
            for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                suffix = "/" if child.is_dir() else ""
                name = child.name + suffix
                if long_form:
                    size = "-" if child.is_dir() else str(child.stat().st_size)
                    rows.append(f"{size:>8}  {name}")
                else:
                    rows.append(name)
            return "\n".join(rows) or "(empty directory)"
        if op == "cat":
            if len(argv) != 2:
                raise ValueError("usage: cat relative-file")
            target = _resolve(root, argv[1], must_exist=True)
            if not target.is_file():
                raise ValueError(f"not a file: {argv[1]}")
            text = target.read_text()
            if len(text) > _MAX_READ_CHARS:
                return text[:_MAX_READ_CHARS] + "\n...[read cap reached]"
            return text
        if op in {"find", "tree"}:
            if len(argv) > 2:
                raise ValueError(f"usage: {op} [relative-directory]")
            target = _resolve(root, argv[1] if len(argv) == 2 else ".", must_exist=True)
            if not target.is_dir():
                return _rel(root, target)
            rows = []
            for child in sorted(target.rglob("*")):
                if "__pycache__" in child.parts:
                    continue
                rows.append(_rel(root, child) + ("/" if child.is_dir() else ""))
            return "\n".join(rows) or "(empty directory)"
        return "ERROR: read-only command must be one of: pwd, ls, cat, find, tree"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


def write_file(root: Path, path: str, content: str) -> str:
    try:
        target = _resolve(root, path)
        relative = _rel(root, target)
        if not _mutable_path(relative):
            raise ValueError(
                "path is not a mutable H2 genome file (runtime and provenance are read-only)"
            )
        if not isinstance(content, str) or "\x00" in content:
            raise ValueError("content must be UTF-8 text without NUL bytes")
        if len(content) > _MAX_WRITE_CHARS:
            raise ValueError(f"content exceeds {_MAX_WRITE_CHARS} characters")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.rstrip() + "\n")
        return f"WROTE {relative} ({target.stat().st_size} bytes)"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


def edit_file(
    root: Path,
    path: str,
    old_text: str = "",
    new_text: str = "",
    append_text: str = "",
) -> str:
    try:
        target = _resolve(root, path, must_exist=True)
        relative = _rel(root, target)
        if not _mutable_path(relative):
            raise ValueError(
                "path is not a mutable H2 genome file (runtime and provenance are read-only)"
            )
        current = target.read_text()
        if append_text:
            if old_text or new_text:
                raise ValueError(
                    "append_text cannot be combined with old_text/new_text"
                )
            if "\x00" in append_text:
                raise ValueError("append_text must be UTF-8 text without NUL bytes")
            if not append_text.strip():
                raise ValueError("append_text must contain non-whitespace text")
            updated = current.rstrip() + "\n\n" + append_text.strip() + "\n"
            if len(updated) > _MAX_WRITE_CHARS:
                raise ValueError(
                    f"updated file exceeds {_MAX_WRITE_CHARS} characters"
                )
            target.write_text(updated)
            return f"APPENDED {relative} ({len(current)} -> {len(updated)} chars)"
        count = current.count(old_text)
        if not old_text:
            raise ValueError("old_text must be non-empty; use write_harness_file to create")
        if count != 1:
            raise ValueError(f"old_text must match exactly once; found {count} matches")
        updated = current.replace(old_text, new_text, 1)
        if len(updated) > _MAX_WRITE_CHARS:
            raise ValueError(f"updated file exceeds {_MAX_WRITE_CHARS} characters")
        target.write_text(updated)
        return f"EDITED {relative} ({len(current)} -> {len(updated)} chars)"
    except (OSError, UnicodeError, ValueError) as exc:
        return f"ERROR: {exc}"


def delete_file(root: Path, path: str) -> str:
    try:
        target = _resolve(root, path, must_exist=True)
        relative = _rel(root, target)
        if not _mutable_path(relative):
            raise ValueError(
                "path is not a mutable H2 genome file (runtime and provenance are read-only)"
            )
        if not target.is_file():
            raise ValueError("delete_harness_file removes one file, not a directory")
        target.unlink()
        return f"DELETED {relative}"
    except (OSError, ValueError) as exc:
        return f"ERROR: {exc}"


def _normalize_agent(agent: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(agent)
    for row in out.get("tools", []):
        if "custom_runtime" not in str(row.get("binding", "")):
            continue
        name = str(row.get("name", ""))
        extra = row.setdefault("extra_kwargs", {})
        extra.pop("py_path", None)  # normalize legacy packages
        extra[hs.RUNTIME_TOOL_PATH_ARG] = f"./custom_tools/{name}.py"
    return out


def _validate_agent_matches_canonical(
    root: Path, canonical_root: Path, errors: List[str],
) -> None:
    try:
        draft = yaml.safe_load((root / "agent.yaml").read_text())
        canonical = yaml.safe_load((canonical_root / "agent.yaml").read_text())
    except Exception as exc:
        errors.append(f"agent.yaml parse error: {exc}")
        return
    if not isinstance(draft, dict):
        errors.append("agent.yaml must contain a mapping")
        return

    # Generated tool source paths may be absolute to different private temp
    # roots.  Normalize only that location; every other field must match the
    # safe canonical compiler output.
    for row in draft.get("tools", []):
        if "custom_runtime" not in str(row.get("binding", "")):
            continue
        name = str(row.get("name", ""))
        extra_kwargs = row.get("extra_kwargs") or {}
        declared = str(
            extra_kwargs.get(hs.RUNTIME_TOOL_PATH_ARG)
            or extra_kwargs.get("py_path", "")
        )
        try:
            resolved = _resolve(root, declared, must_exist=True) \
                if not Path(declared).is_absolute() else Path(declared).resolve()
            expected = (root / "custom_tools" / f"{name}.py").resolve()
            if resolved != expected:
                errors.append(
                    f"agent.yaml tool {name!r} py_path must point to "
                    f"custom_tools/{name}.py inside this H2"
                )
        except Exception as exc:
            errors.append(f"agent.yaml tool {name!r} has invalid py_path: {exc}")
    if _normalize_agent(draft) != _normalize_agent(canonical):
        # SCAFFOLD-006: every SUPPORTED edit survives the spec parse and is
        # present in the canonical agent.yaml; whatever the draft has beyond
        # that is unexpressible in the genome.  Overwrite the draft with the
        # canonical file — reverting exactly the unsupported edits — instead
        # of failing the slot (v80d lost two slots to this).
        try:
            (root / "agent.yaml").write_text(
                (canonical_root / "agent.yaml").read_text())
            return
        except OSError:
            pass
        errors.append(
            "agent.yaml contains unsupported or inconsistent edits. Editable "
            "semantics are sampling, max_iterations, supported middleware params, "
            "and the canonical tool/skill/middleware mount lists."
        )


def _validate_component_files(root: Path, effective: Dict[str, Any], errors: List[str]) -> None:
    inventory = hs.generated_component_inventory(effective)

    tool_names = set(inventory["new_tools"])
    custom_dir = root / "custom_tools"
    actual_tool_code = {
        path.stem for path in custom_dir.glob("*.py")
        if path.name != "__init__.py"
    } if custom_dir.exists() else set()
    if actual_tool_code != tool_names:
        errors.append(
            "custom_tools files must exactly match mounted generated tools: "
            f"files={sorted(actual_tool_code)}, mounted={sorted(tool_names)}"
        )

    tool_schema_names = {
        path.name[:-len(".tool.yaml")]
        for path in (root / "tools").glob("*.tool.yaml")
    } if (root / "tools").exists() else set()
    custom_schemas = tool_schema_names - _CORE_TOOL_NAMES
    if custom_schemas != tool_names:
        errors.append(
            "custom tool schemas must exactly match mounted generated tools: "
            f"schemas={sorted(custom_schemas)}, mounted={sorted(tool_names)}"
        )
    for path in (root / "tools").glob("*.tool.yaml"):
        try:
            doc = yaml.safe_load(path.read_text()) or {}
            expected_name = path.name[:-len(".tool.yaml")]
            if set(doc) != {"type", "name", "description", "input_schema"}:
                errors.append(f"{path.relative_to(root)} has unsupported/missing keys")
            if doc.get("type") != "tool" or doc.get("name") != expected_name:
                errors.append(f"{path.relative_to(root)} type/name does not match its filename")
        except Exception as exc:
            errors.append(f"cannot parse {path.relative_to(root)}: {exc}")

    skill_names = set(inventory["new_skills"])
    actual_skills = {
        path.parent.name
        for path in (root / "skills").glob("*/SKILL.md")
        if path.parent.name != "discovery-optimization"
    } if (root / "skills").exists() else set()
    if actual_skills != skill_names:
        errors.append(
            "skill files must exactly match mounted generated skills: "
            f"files={sorted(actual_skills)}, mounted={sorted(skill_names)}"
        )
    for path in (root / "skills").glob("*/SKILL.md"):
        try:
            text = path.read_text()
            match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
            if not match:
                errors.append(f"{path.relative_to(root)} needs YAML frontmatter")
                continue
            frontmatter = yaml.safe_load(match.group(1)) or {}
            if set(frontmatter) != {"name", "description"}:
                errors.append(
                    f"{path.relative_to(root)} frontmatter must contain exactly "
                    "name and description"
                )
            if frontmatter.get("name") != path.parent.name:
                errors.append(
                    f"{path.relative_to(root)} name does not match its directory"
                )
        except Exception as exc:
            errors.append(f"cannot parse {path.relative_to(root)}: {exc}")

    middleware_names = set(inventory["new_middlewares"])
    mw_dir = root / "middlewares"
    actual_middleware_code = {
        path.stem for path in mw_dir.glob("*.py")
        if path.name not in _FIXED_MIDDLEWARE_FILES
    } if mw_dir.exists() else set()
    descriptors = {
        path.name[:-len(".middleware.yaml")]
        for path in mw_dir.glob("*.middleware.yaml")
    } if mw_dir.exists() else set()
    if actual_middleware_code != middleware_names or descriptors != middleware_names:
        errors.append(
            "middleware code/descriptors must exactly match mounted generated "
            f"middlewares: code={sorted(actual_middleware_code)}, "
            f"descriptors={sorted(descriptors)}, mounted={sorted(middleware_names)}"
        )
    for path in mw_dir.glob("*.middleware.yaml") if mw_dir.exists() else []:
        try:
            doc = yaml.safe_load(path.read_text()) or {}
            expected_name = path.name[:-len(".middleware.yaml")]
            if set(doc) != {"name", "hook", "description"}:
                errors.append(
                    f"{path.relative_to(root)} must contain exactly name, hook, description"
                )
            if doc.get("name") != expected_name or doc.get("hook") != "before_model":
                errors.append(
                    f"{path.relative_to(root)} name/hook does not match the mounted middleware"
                )
        except Exception as exc:
            errors.append(f"cannot parse {path.relative_to(root)}: {exc}")


def _audit_generated_tools(
    root: Path, effective: Dict[str, Any], errors: List[str],
) -> List[Dict[str, Any]]:
    """Gate and self-test mounted tool code without modifying the candidate."""

    from outer.reviewer.reviewer import review_tool_code

    audit: List[Dict[str, Any]] = []
    tools_by_name = {
        str(row.get("name")): row
        for row in effective.get("new_tools", [])
        if isinstance(row, dict) and row.get("name")
    }
    for name in hs.generated_component_inventory(effective)["new_tools"]:
        path = root / "custom_tools" / f"{name}.py"
        if not path.is_file():
            continue
        outcome = review_tool_code(
            path.read_text(),
            input_schema=(tools_by_name.get(name) or {}).get("input_schema"),
            repair_fn=None,
            max_rounds=0,
            selftest_timeout=10.0,
        )
        row = {
            "kind": "tool", "name": name, "ok": bool(outcome.ok),
            "mutated_after_h1": False, "error": outcome.final_error,
            "history": outcome.history,
        }
        audit.append(row)
        if not outcome.ok:
            errors.append(
                f"generated tool {name!r} failed validate_harness: "
                f"{outcome.final_error or 'unknown gate/self-test failure'}"
            )
    return audit


def _extract_hook_def(source: str, hook_name: str):
    """Return the source of ``def <hook_name>(...)`` if it parses, else None."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == hook_name:
            return ast.get_source_segment(source, node)
    # also accept a hook nested inside the rewritten wrapper class
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == hook_name:
            seg = ast.get_source_segment(source, node)
            if seg is not None:
                import textwrap
                return textwrap.dedent(seg)
    return None


def _wrap_user_hook(name: str, hook_name: str, hook_src: str) -> str:
    """Rebuild the canonical wrapper around an extracted user hook."""
    from outer.compiling.materialize import _MW_WRAPPER
    return _MW_WRAPPER.format(user_code=hook_src.strip(), hook=hook_name,
                              name_repr=repr(name), hook_repr=repr(hook_name))


def _validate_canonical_component_files(
    root: Path, canonical_root: Path, effective: Dict[str, Any], errors: List[str],
) -> None:
    """Reject edits that parsing would otherwise silently canonicalize away."""

    mw_specs = {str(m.get("name")): m
                for m in (effective.get("new_middlewares") or [])
                if isinstance(m, dict)}
    for name in hs.generated_component_inventory(effective)["new_middlewares"]:
        hook_name = str((mw_specs.get(name) or {}).get("hook") or "before_model")
        draft_path = root / "middlewares" / f"{name}.py"
        canonical_path = canonical_root / "middlewares" / f"{name}.py"
        try:
            draft = draft_path.read_text().strip()
            canonical = canonical_path.read_text().strip()
        except OSError as exc:
            errors.append(f"cannot compare generated middleware {name!r}: {exc}")
            continue
        has_wrapper = "class GeneratedMiddleware" in draft
        has_sentinels = (
            "# --USER-HOOK-START--" in draft
            and "# --USER-HOOK-END--" in draft
        )
        if has_wrapper and not has_sentinels:
            errors.append(
                f"middleware {name!r} wrapper is missing USER-HOOK sentinels"
            )
        elif has_sentinels and draft != canonical:
            # SCAFFOLD-003: a rewritten wrapper is normalised, not refused.
            # Measured failure mode: the proposer rewrites the WHOLE file
            # (wrapper included) around a perfectly good hook, then burns its
            # remaining turns fighting this error.  If a def for the declared
            # hook parses inside the draft, extract it and rewrite the file
            # as canonical wrapper + that hook.
            hook_src = _extract_hook_def(draft, hook_name)
            if hook_src is not None:
                draft_path.write_text(
                    _wrap_user_hook(name, hook_name, hook_src))
            else:
                errors.append(
                    f"middleware {name!r} changed code outside its USER-HOOK "
                    "region and no def for its hook parses in the file; edit "
                    "between the sentinels or write a plain def "
                    f"{hook_name}(hook_input) file"
                )


def _drop_implementationless_tool_mounts(root: Path) -> None:
    """SCAFFOLD-004: unmount generated tools that have no code file.

    Measured failure mode: the proposer mounts step_initializer in
    agent.yaml, never writes custom_tools/step_initializer.py, and the
    whole workspace fails parse.  The mount is dead weight — dropping it
    banks whatever else the candidate built (a forced slot still holds its
    seeded starter, and a component-only slot without a component is still
    refused at submit)."""
    agent_p = root / "agent.yaml"
    try:
        agent = yaml.safe_load(agent_p.read_text()) or {}
    except Exception:
        return
    mounts = agent.get("tools") or []
    kept, dropped = [], []
    for m in mounts:
        if isinstance(m, dict) and "custom_runtime" in str(m.get("binding", "")):
            extra = m.get("extra_kwargs") or {}
            rel = str(extra.get(hs.RUNTIME_TOOL_PATH_ARG)
                      or extra.get("py_path") or "")
            if rel and not (root / rel).is_file():
                dropped.append(str(m.get("name")))
                continue
        kept.append(m)
    if dropped:
        agent["tools"] = kept
        agent_p.write_text(yaml.safe_dump(agent, sort_keys=False))
        # the schema file is KEPT: _complete_orphan_tools may re-mount the
        # tool once its code exists, and regenerating an empty schema would
        # make the tool uncallable with arguments (jsonschema refuses them)


def _complete_orphan_middlewares(root: Path) -> None:
    """SCAFFOLD-004: complete an orphan middleware file mechanically.

    Measured failure mode: the proposer writes family_guard.py with a good
    hook and runs out of turns before the descriptor, the mount, and the
    prompt declaration.  Runs BEFORE the spec parse so every later
    consistency check sees the completed state."""
    mw_dir = root / "middlewares"
    if not mw_dir.is_dir():
        return
    agent_p = root / "agent.yaml"
    try:
        agent = yaml.safe_load(agent_p.read_text()) or {}
    except Exception:
        return
    mounts = agent.get("middlewares") or []
    mounted = {str(m.get("import", "")).split(".")[-1].split(":")[0]
               for m in mounts if isinstance(m, dict)}
    changed = False
    for path in sorted(mw_dir.glob("*.py")):
        name = path.stem
        if path.name in _FIXED_MIDDLEWARE_FILES or name in mounted:
            continue
        hook_src = _extract_hook_def(path.read_text(), "before_model")
        if hook_src is not None:
            try:
                from outer.safety.static_gates import check_middleware_code
                ok, _errs = check_middleware_code(hook_src, "before_model")
            except Exception:
                ok = False
            if not ok:
                # ungated middleware is spliced in at module level and called
                # every model turn; its tool sibling has always been gated
                hook_src = None
        if hook_src is None:
            # SCAFFOLD-005: an orphan middleware with no extractable hook can
            # never mount — delete it so the rest of the work validates.
            path.unlink()
            desc = mw_dir / f"{name}.middleware.yaml"
            if desc.is_file():
                desc.unlink()
            changed = True
            continue
        path.write_text(_wrap_user_hook(name, "before_model", hook_src))
        (mw_dir / f"{name}.middleware.yaml").write_text(yaml.safe_dump({
            "name": name, "hook": "before_model",
            "description": f"Generated middleware {name}.",
        }, sort_keys=False))
        mounts.insert(0, {"import": f"middlewares.{name}:GeneratedMiddleware",
                          "params": {}})
        changed = True
        prompt_p = root / "prompt.md"
        if name not in prompt_p.read_text():
            prompt_p.write_text(
                prompt_p.read_text().rstrip("\n")
                + f"\n\n# Generated Middleware: {name}\nRuns automatically "
                "before each model call; no tool call is needed.\n")
    if changed:
        agent["middlewares"] = mounts
        agent_p.write_text(yaml.safe_dump(agent, sort_keys=False))


def _complete_orphan_tools(root: Path) -> None:
    """SCAFFOLD-004b: mount an orphan custom_tools/<name>.py mechanically.

    Symmetric to _complete_orphan_middlewares — final2's r0 k1 wrote
    structured_init.py, never mounted it, and the workspace failed the
    exact-match check.  Only files that pass the static tool gate are
    completed; anything else keeps failing loudly."""
    ct_dir = root / "custom_tools"
    if not ct_dir.is_dir():
        return
    agent_p = root / "agent.yaml"
    try:
        agent = yaml.safe_load(agent_p.read_text()) or {}
    except Exception:
        return
    mounts = agent.get("tools") or []
    mounted = {str(m.get("name")) for m in mounts if isinstance(m, dict)}
    changed = False
    for path in sorted(ct_dir.glob("*.py")):
        name = path.stem
        if name in mounted:
            continue
        try:
            from outer.safety.static_gates import check_tool_code
            ok, _errs = check_tool_code(path.read_text())
        except Exception:
            ok = False
        if not ok:
            # SCAFFOLD-005: a gate-failing orphan tool can never mount —
            # delete it (and its spec) so the rest of the work validates.
            path.unlink()
            dead_spec = root / "tools" / f"{name}.tool.yaml"
            if dead_spec.is_file():
                dead_spec.unlink()
            continue
        spec_p = root / "tools" / f"{name}.tool.yaml"
        if not spec_p.is_file():
            spec_p.write_text(yaml.safe_dump({
                "type": "tool", "name": name,
                "description": f"Generated tool {name}.",
                "input_schema": {"type": "object", "properties": {},
                                 "additionalProperties": False},
            }, sort_keys=False))
        mounts.append({
            "name": name, "yaml_path": f"./tools/{name}.tool.yaml",
            "binding": "inner.harness.tools.custom_runtime:custom_tool",
            "extra_kwargs": {"py_path": f"./custom_tools/{name}.py"},
        })
        changed = True
        prompt_p = root / "prompt.md"
        if name not in prompt_p.read_text():
            prompt_p.write_text(
                prompt_p.read_text().rstrip("\n")
                + f"\n\n# Generated Tool: {name}\nDecide its trigger before "
                "the first edit; call it when the trigger holds.\n")
    if changed:
        agent["tools"] = mounts
        agent_p.write_text(yaml.safe_dump(agent, sort_keys=False))


def _complete_orphan_skills(root: Path) -> None:
    """SCAFFOLD-005: mount an orphan skills/<slug>/SKILL.md mechanically."""
    sk_dir = root / "skills"
    if not sk_dir.is_dir():
        return
    agent_p = root / "agent.yaml"
    try:
        agent = yaml.safe_load(agent_p.read_text()) or {}
    except Exception:
        return
    mounts = [str(s) for s in (agent.get("skills") or [])]
    mounted = {m.rstrip("/").split("/")[-1] for m in mounts}
    changed = False
    for skill_md in sorted(sk_dir.glob("*/SKILL.md")):
        slug = skill_md.parent.name
        if slug in mounted:
            continue
        mounts.append(f"./skills/{slug}")
        changed = True
        prompt_p = root / "prompt.md"
        if slug not in prompt_p.read_text():
            prompt_p.write_text(
                prompt_p.read_text().rstrip("\n")
                + f"\n\n# Generated Skill: {slug}\nEnact the {slug} skill "
                "before the first edit.\n")
    if changed:
        agent["skills"] = mounts
        agent_p.write_text(yaml.safe_dump(agent, sort_keys=False))


def _add_missing_prompt_declarations(root: Path) -> None:
    """SCAFFOLD-006: declare mounted generated components the prompt omits.

    Mirror image of the stale-declaration strip — v80d lost three slots to
    'system_prompt does not name current tools/middlewares' after the model
    mounted a component and never declared it."""
    prompt_p = root / "prompt.md"
    try:
        text = prompt_p.read_text()
        agent = yaml.safe_load((root / "agent.yaml").read_text()) or {}
    except Exception:
        return
    additions = []
    for m in (agent.get("tools") or []):
        if isinstance(m, dict) and "custom_runtime" in str(m.get("binding", "")):
            name = str(m.get("name"))
            if name and name not in text:
                additions.append(
                    f"\n# Generated Tool: {name}\nDecide its trigger before "
                    "the first edit; call it when the trigger holds.\n")
    fixed = {f[:-3] for f in _FIXED_MIDDLEWARE_FILES if f.endswith(".py")}
    for m in (agent.get("middlewares") or []):
        imp = str(m.get("import", "")) if isinstance(m, dict) else ""
        if imp.startswith("middlewares."):
            name = imp.split(".")[-1].split(":")[0]
            if name and name not in fixed and name not in text:
                additions.append(
                    f"\n# Generated Middleware: {name}\nRuns automatically "
                    "before each model call; no tool call is needed.\n")
    for s in (agent.get("skills") or []):
        slug = str(s).rstrip("/").split("/")[-1]
        if slug and slug != "discovery-optimization" and slug not in text:
            additions.append(
                f"\n# Generated Skill: {slug}\nEnact the {slug} skill "
                "before the first edit.\n")
    if additions:
        cap = hs._TEXT_FIELDS.get("system_prompt", 10000)
        merged = text.rstrip("\n") + "\n"
        for add in additions:
            if len(merged) + len(add) > cap:
                break          # an over-cap prompt is an INVALID workspace
            merged += add
        prompt_p.write_text(merged)


def _strip_stale_prompt_declarations(root: Path) -> None:
    """SCAFFOLD-005: drop prompt declarations for components that no longer
    exist anywhere (no file, no mount) — e.g. a declared tool whose dead
    mount was dropped and whose file was never written."""
    prompt_p = root / "prompt.md"
    try:
        text = prompt_p.read_text()
    except OSError:
        return
    try:
        agent = yaml.safe_load((root / "agent.yaml").read_text()) or {}
    except Exception:
        return
    alive = {str(m.get("name")) for m in (agent.get("tools") or [])
             if isinstance(m, dict)}
    alive |= {str(m.get("import", "")).split(".")[-1].split(":")[0]
              for m in (agent.get("middlewares") or []) if isinstance(m, dict)}
    alive |= {str(s).rstrip("/").split("/")[-1]
              for s in (agent.get("skills") or [])}
    alive |= {f.stem for f in (root / "custom_tools").glob("*.py")}         if (root / "custom_tools").is_dir() else set()
    lines = text.splitlines()
    out: List[str] = []
    skip = False
    import re
    for line in lines:
        m = re.match(r"#+\s*Generated (Tool|Middleware|Skill):\s*(\S+)", line)
        if m:
            # NORMALISE the captured name the way the genome's declaration
            # checker does: a proposer writing "# Generated Tool: `route_check`"
            # (or with a trailing period) declares a LIVE component, and the
            # raw capture made it look dead — measured: the heading plus the
            # proposer's four-line usage contract were deleted and replaced
            # with boilerplate, inside system_prompt, so the paired control
            # measured the normaliser's prompt instead of the proposal's.
            name = m.group(2).strip("`*_.,:;()[]\"'")
            skip = name not in alive
            if skip:
                while out and not out[-1].strip():
                    out.pop()
                continue
        elif skip and (not line.strip() or line.startswith("#")):
            # a dead block ends at its blank line, not at the next heading
            skip = False
            if line.startswith("#"):
                out.append(line)
            continue
        if not skip:
            out.append(line)
    new_text = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if new_text != text:
        prompt_p.write_text(new_text)


def validate_workspace(root: Path, base_spec: Dict[str, Any]) -> WorkspaceCheck:
    root = Path(root).resolve()
    errors: List[str] = []
    for required in ("agent.yaml", "prompt.md"):
        if not (root / required).is_file():
            errors.append(f"missing required H2 file: {required}")
    if errors:
        return WorkspaceCheck(False, errors)
    _drop_implementationless_tool_mounts(root)
    _complete_orphan_middlewares(root)
    _complete_orphan_tools(root)
    _complete_orphan_skills(root)
    _strip_stale_prompt_declarations(root)
    _add_missing_prompt_declarations(root)

    try:
        # meta.json/component_manifest.json describe the accepted parent and are
        # deliberately read-only while H1 edits.  Workspace truth is the live
        # mount/file graph; provenance is regenerated only after acceptance.
        raw_effective = hs.read_base_spec(root, verify_provenance=False)
    except Exception as exc:
        return WorkspaceCheck(False, [f"cannot parse H2 workspace: {exc}"])

    partial = hs.diff_to_partial(raw_effective, base_spec)
    differs, changed = hs.differs_from_base(raw_effective, base_spec)
    _validate_component_files(root, raw_effective, errors)
    component_audit = _audit_generated_tools(root, raw_effective, errors)
    errors.extend(hs.component_prompt_issues(raw_effective))
    if not differs:
        errors.append("workspace is identical to the current H2 (no-op)")
        return WorkspaceCheck(
            False, errors,
            partial=partial, effective=raw_effective,
            component_audit=component_audit,
        )

    validation = hs.parse_and_validate(
        yaml.safe_dump(partial, sort_keys=False, allow_unicode=True, width=100)
    )
    if not validation.valid:
        return WorkspaceCheck(
            False, errors + validation.errors, partial=partial,
            effective=raw_effective, changed_fields=changed,
            component_audit=component_audit,
        )
    effective = hs.merge_with_base(validation.spec or {}, base_spec)
    if hs.canonical_json(effective) != hs.canonical_json(raw_effective):
        errors.append(
            "workspace contains edits that cannot be represented by the H2 genome"
        )

    prompt = str(effective.get("system_prompt", ""))
    current_inventory = hs.generated_component_inventory(effective)
    base_inventory = hs.generated_component_inventory(base_spec)
    for field in hs.GENERATED_COMPONENT_FIELDS:
        removed = set(base_inventory[field]) - set(current_inventory[field])
        for name in removed:
            if name in prompt:
                errors.append(
                    f"system_prompt still advertises removed component {name!r}"
                )

    with tempfile.TemporaryDirectory(prefix="h2_canonical_") as td:
        canonical_root = Path(td) / "h2"
        try:
            materialize(
                effective,
                canonical_root,
                meta={"effective": effective},
            )
        except Exception as exc:
            errors.append(f"canonical H2 compilation failed: {exc}")
        else:
            _validate_agent_matches_canonical(root, canonical_root, errors)
            _validate_canonical_component_files(
                root, canonical_root, effective, errors
            )

    return WorkspaceCheck(
        not errors,
        errors=errors,
        partial=validation.spec,
        effective=effective,
        changed_fields=changed,
        component_audit=component_audit,
    )
