"""Outer-round orchestrator CLI (instance-wise).

Per round, for EACH task instance tau in the task list (plan.md §2.2):
K H1-agent runs conditioned on tau -> K candidate H2 packages for tau ->
each rolled out once on tau by the inner loop -> per-task GRPO group.

Stages (run inside the serving job; see scripts/outer_round.sbatch):

  propose  for every (task, k): run H1 against M_phi -> validate -> materialize
           <round_dir>/tasks/<task_id>/candNN/ + round.json/prompts/trajectories.
  collect  after rollouts: per-task rewards + group advantages ->
           grpo_batch.jsonl (K x n_tasks rows) + round_summary.json +
           next_bases.json (per-task best harness -> next round's bases).

Bases registry: {task_id: {"package": dir, "score": s}} — which H2 package is
the current best for each task and what it scored. Round 1 defaults to the
initial hand-written H2 + results/baseline_h2_20ev.json scores; later rounds
pass --bases-file <prev_round>/next_bases.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from outer.proposing import proposer_io as pio, propose as pp  # noqa: E402
from outer.genome import harness_spec as hs  # noqa: E402
from outer.reward import rewards as rw  # noqa: E402
from outer.reward.program_ratchet import suppress_task_training, update_program_ratchet  # noqa: E402
from outer.compiling.materialize import materialize, INNER_HARNESS  # noqa: E402
from inner.runtime.package_hash import h2_sha256  # noqa: E402

REPO = _SRC.parent
BASELINE_JSON = Path(os.environ.get(
    "SAH_BASELINE_JSON", str(REPO / "data" / "baseline_h2_20ev.json")))

DEFAULT_TASKS = [
    "eft__math__circle_packing",
    "eft__math__hadamard_maximal_det",
    "eft__math__erdos_min_overlap",
    "adrs__prism",
    "adrs__txn_scheduling",
    "adrs__eplb",
    "eft__algotune__convolve2d_full_fill",
    "eft__algotune__psd_cone_projection",
]


def _load_bases(bases_file: str | None, tasks: list) -> Dict[str, Dict[str, Any]]:
    """task_id -> {package, score, seed_score}.

    COLDSTART-001: a task absent from the static baseline file (any newly
    onboarded task — the next phase runs CODE tasks) starts from the fixed
    H2 with score 0.0 and a ``cold_start`` marker instead of KeyError-ing
    the round.  The number is honest within one round: round 0's paired
    controls are rollouts of this very base, and BASEREFRESH-001 replaces
    the placeholder with their mean before round 1 inherits it.
    """
    try:
        static = json.loads(BASELINE_JSON.read_text())["baseline"]
    except (OSError, ValueError, KeyError):
        print(f"[bases] {BASELINE_JSON} unreadable — all tasks cold start")
        static = {}

    def _static_entry(t: str) -> Dict[str, Any]:
        if t in static:
            return {"package": str(INNER_HARNESS),
                    "score": static[t]["h2_best"]}
        print(f"[bases] {t}: not in {BASELINE_JSON.name} — cold start "
              "(score 0.0 until round-0 controls refresh it)")
        return {"package": str(INNER_HARNESS), "score": 0.0,
                "cold_start": True}

    if bases_file:
        bases = json.loads(Path(bases_file).read_text())
    else:
        bases = {t: _static_entry(t) for t in tasks}
    for t in tasks:
        if t not in bases:
            bases[t] = _static_entry(t)
        bases[t].setdefault("seed_score", static.get(t, {}).get("seed", 0.0))
    return bases


def _input_file_provenance(path_value: str | None) -> Dict[str, Any]:
    """Hash a mutable cross-round input at the moment this round consumes it."""
    if not path_value:
        return {"path": None, "exists": False, "sha256": None, "bytes": 0}
    path = Path(path_value)
    if not path.is_file():
        return {"path": str(path), "exists": False, "sha256": None, "bytes": 0}
    raw = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _replica_base_urls(base_url: str, n_replicas: int) -> list[str]:
    """Expand a concrete base endpoint across consecutive local ports."""
    if n_replicas <= 0:
        return [base_url]
    parsed = urlsplit(base_url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(
            "--base-url must include a hostname and port when --n-replicas > 0"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("userinfo is not supported in replicated --base-url")
    last_port = parsed.port + n_replicas - 1
    if last_port > 65535:
        raise ValueError("replica port range exceeds 65535")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    return [
        urlunsplit((
            parsed.scheme,
            f"{host}:{parsed.port + index}",
            parsed.path,
            parsed.query,
            parsed.fragment,
        ))
        for index in range(n_replicas)
    ]


def cmd_propose(args) -> None:
    from inner.tasks.eft_task import get_task  # tasks registry (spec + seed program)

    round_dir = Path(args.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    bases = _load_bases(args.bases_file, args.tasks)
    feedback_input = _input_file_provenance(
        getattr(args, "feedback_file", None)
    )
    program_input = _input_file_provenance(
        getattr(args, "seed_programs_file", None)
    )
    bases_input = _input_file_provenance(getattr(args, "bases_file", None))
    base_urls = _replica_base_urls(args.base_url, args.n_replicas)

    # per-task context: base spec (that task's current best harness) + user message
    inherited = {}
    logical_index = int(os.environ.get("LOGICAL_ROUND_INDEX", "0") or 0)
    if os.environ.get("SAH_FIXED_INFERENCE_SLOTS", "0") == "1" \
            and logical_index > 0 \
            and not getattr(args, "seed_programs_file", None):
        raise RuntimeError(
            "canonical post-cold round requires --seed-programs-file"
        )
    if getattr(args, "seed_programs_file", None):
        try:
            inherited = json.loads(Path(args.seed_programs_file).read_text())
            if not isinstance(inherited, dict):
                raise TypeError("seed-program registry must be a JSON object")
        except Exception as e:
            if os.environ.get("SAH_FIXED_INFERENCE_SLOTS", "0") == "1" \
                    and logical_index > 0:
                raise RuntimeError(
                    "canonical post-cold round requires its task-local "
                    "seed-program registry"
                ) from e
            print(f"[propose] WARNING: seed-programs-file unreadable ({e}); using task seeds")
            inherited = {}
    # Rollouts consume this immutable round-local copy, never the mutable
    # workspace registry that the collector will update after the batch.
    seed_snapshot = round_dir / "seed_programs_in.json"
    seed_snapshot.write_text(
        json.dumps(inherited, indent=2, ensure_ascii=False) + "\n"
    )
    snapshot_raw = seed_snapshot.read_bytes()
    program_input["snapshot_path"] = str(seed_snapshot.resolve())
    program_input["snapshot_sha256"] = hashlib.sha256(snapshot_raw).hexdigest()
    program_input["snapshot_bytes"] = len(snapshot_raw)

    # Anti-tampering: verify the task texts about to be served against the
    # pinned registry (fail-closed under SAH_TASK_TEXT_ENFORCE=1).
    from outer.rounds.task_text_registry import verify_task_texts
    _tasks_by_id = {tid: get_task(tid) for tid in args.tasks}
    task_text_provenance = verify_task_texts(_tasks_by_id)
    (round_dir / "task_text_provenance.json").write_text(
        json.dumps(task_text_provenance, indent=1)
    )
    for _tid, _row in task_text_provenance.get("tasks", {}).items():
        if _row.get("status") == "MISMATCH":
            print(f"[task-text] WARNING: {_tid} differs from the pinned registry")

    ctx: Dict[str, Dict[str, Any]] = {}
    for tid in args.tasks:
        task = _tasks_by_id[tid]
        # BEAM-001: with CURVE_INHERIT_BEAM=B>1 the task carries B parent
        # packages; slots are dealt round-robin across them so a runner-up
        # lineage (e.g. the only component-carrying candidate) survives a
        # round instead of being deleted by strict single inheritance.
        beam_entries = list(bases[tid].get("beam") or [])
        if not beam_entries:
            beam_entries = [{"package": bases[tid]["package"],
                             "score": bases[tid]["score"], "from": "incumbent"}]
        base_spec = hs.read_base_spec(Path(bases[tid]["package"]))
        beam_specs = [hs.read_base_spec(Path(e["package"])) for e in beam_entries]
        ent = inherited.get(tid)
        if ent:  # inheritance: rollouts start from the current best program
            seed_prog = ent["program"] if isinstance(ent, dict) else ent
            seed_sc = float(ent.get("score", bases[tid]["score"])) if isinstance(ent, dict) \
                else float(bases[tid]["score"])
        else:
            seed_prog, seed_sc = task.initial_program, float(bases[tid]["seed_score"])
        fb_text = ""
        fb = None
        analysis_enabled = os.environ.get("SAH_ANALYSIS", "0") == "1"
        analysis_required = os.environ.get("SAH_ANALYSIS_REQUIRED", "0") == "1"
        if analysis_required and not analysis_enabled:
            raise RuntimeError(
                "SAH_ANALYSIS_REQUIRED=1 requires SAH_ANALYSIS=1"
            )
        analysis_meta = {
            "enabled": analysis_enabled,
            "required_after_cold_round": analysis_required,
            "feedback_available": False,
            "brief_attached": False,
            "model_calls": 0,
            "specialists": [],
            "feedback_input": feedback_input,
        }
        if getattr(args, "feedback_file", None):
            try:
                fb = json.loads(Path(args.feedback_file).read_text()).get(tid)
                if fb:
                    fb_text = pio.render_feedback(fb)
                    analysis_meta["feedback_available"] = True
            except Exception as e:
                print(f"[propose] WARNING: feedback-file unreadable ({e})")
        # optional analysis brief (campaign_config: analysis.enabled). Two
        # read-only specialists on the SAME frozen M0 distill the feedback into a
        # bounded, leak-guarded brief appended to the proposer message. Legacy
        # runs fail open; the canonical context arm sets SAH_ANALYSIS_REQUIRED=1
        # and fails closed after the cold round.
        if analysis_enabled and fb:
            try:
                from outer.rounds import analysis as an
                # LEAK GUARD: analysis MUST run on the frozen M0, never the
                # trained proposer M_phi. Under split serving base_urls[0] is
                # M_phi; the worker exports SAH_ANALYSIS_BASE_URL pointing at a
                # frozen replica. Without split serving base_urls[0] is already
                # the frozen base, so that fallback is also frozen M0.
                an_url = os.environ.get("SAH_ANALYSIS_BASE_URL") or base_urls[0]
                chat = an.make_chat_fn(an_url, args.model)
                brief = an.run_analysis(
                    fb, chat=chat,
                    want_perf=os.environ.get("SAH_ANALYSIS_PERF", "1") == "1",
                    want_design=os.environ.get("SAH_ANALYSIS_DESIGN", "1") == "1")
                if brief:
                    fb_text += brief
                    analysis_meta.update({
                        "brief_attached": True,
                        "model_calls": 2,
                        "specialists": ["performance", "design"],
                    })
                    print(f"[propose] {tid}: analysis brief attached ({len(brief)} chars)")
                elif analysis_required:
                    raise RuntimeError(
                        f"{tid}: required post-cold analyzer produced no brief"
                    )
            except Exception as e:
                if analysis_required:
                    raise RuntimeError(
                        f"{tid}: required post-cold analyzer failed"
                    ) from e
                print(f"[propose] WARNING: analysis stage skipped ({e})")
        # H1 v2.4 campaign evidence: causally-futile proposal signatures
        # (tried >=2 times, |paired delta| always < 1e-4), the plateau streak,
        # and the incumbent prompt's cap headroom.  Best-effort: any error
        # degrades to an empty section, never a failed round.
        campaign_context = ""
        try:
            import collections
            rounds_root = round_dir.parent
            sig_deltas: dict = collections.defaultdict(list)
            for prev in sorted(rounds_root.glob("round0*")):
                if prev == round_dir:
                    continue
                gb = prev / "grpo_batch.jsonl"
                if not gb.is_file():
                    continue
                deltas_by_k = {}
                for line in gb.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("task_id") != tid:
                        continue
                    if row.get("causal_delta") is not None:
                        deltas_by_k[int(row["k"])] = abs(
                            float(row["causal_delta"]))
                for md in sorted((prev / "tasks" / tid).glob("cand*/meta.json")):
                    try:
                        meta_c = json.loads(md.read_text())
                    except Exception:
                        continue
                    k_c = meta_c.get("k")
                    if k_c is None or int(k_c) not in deltas_by_k:
                        continue
                    # component-level signature only: prompt edits are a
                    # class, not a repeatable proposal; named components are.
                    sig = ";".join(sorted(
                        str(f) for f in meta_c.get("changed_fields") or []
                        if not str(f).startswith("system_prompt")
                        and not str(f).startswith("new_tools.quick_probe")
                    ))
                    if sig:
                        sig_deltas[sig].append(deltas_by_k[int(k_c)])
            futile = sorted(
                sig for sig, ds in sig_deltas.items()
                if len(ds) >= 2 and max(ds) < 1e-4
            )
            streak = 0
            curve_path = rounds_root.parent / "curve.jsonl"
            if curve_path.is_file():
                for line in reversed(curve_path.read_text().splitlines()):
                    if not line.strip():
                        continue
                    d = json.loads(line).get("best_causal_delta")
                    if d is not None and abs(float(d)) < 1e-4:
                        streak += 1
                    else:
                        break
            prompt_path = Path(bases[tid]["package"]) / "prompt.md"
            headroom = None
            if prompt_path.is_file():
                headroom = max(0, 10000 - len(prompt_path.read_text()))
            parts = []
            if futile:
                parts.append(
                    "- CAUSALLY FUTILE (tried repeatedly, paired delta always "
                    "~0 — do NOT re-propose): " + "; ".join(futile[:6]))
            if streak >= 2:
                parts.append(
                    f"- SATURATED: the incumbent construction family has "
                    f"plateaued for {streak} straight rounds (every candidate "
                    "AND its control score identically). Marginal edits are "
                    "verified worthless; only a STRUCTURALLY different "
                    "construction can move this score.")
            # DIM6-001: the proposer was blind to how the incumbent's own
            # rollouts behaved (240/240 briefings carried scores only).  The
            # previous round's paired CONTROL rollouts are exactly the
            # incumbent's behaviour, so summarise their ledgers: budget
            # actually spent, tool usage, and stop reasons.
            behaviour = []
            try:
                prev_rounds = sorted(rounds_root.glob("round0*"))
                prev = prev_rounds[-2] if len(prev_rounds) >= 2 else None
                if prev is not None:
                    evals, edits, probes, n, stops = [], [], [], 0, {}
                    for res in (prev / "paired_controls" / tid).glob(
                            "cand*/*/results/*.json"):
                        row = json.loads(res.read_text())
                        rows_ = row if isinstance(row, list) else [row]
                        for rr in rows_:
                            if rr.get("task_id") != tid:
                                continue
                            led = rr.get("ledger") or {}
                            n += 1
                            evals.append(int(led.get("evaluator_calls") or 0))
                            edits.append(int(led.get("edit_calls") or 0))
                            probes.append(int(led.get("probe_calls") or 0))
                            sr = str(rr.get("stop_reason"))
                            stops[sr] = stops.get(sr, 0) + 1
                    if n:
                        behaviour.append(
                            "- INCUMBENT EXECUTOR BEHAVIOUR (last round's "
                            f"{n} control rollouts): evaluator calls "
                            f"{sum(evals)/n:.1f}/{20} used, edits "
                            f"{sum(edits)/n:.1f}, probe calls "
                            f"{sum(probes)/n:.1f}, stop reasons "
                            + ", ".join(f"{k}x{v}" for k, v in stops.items())
                            + ". Unused budget and zero probing are "
                            "opportunities a component could exploit."
                        )
            except Exception:
                pass
            parts.extend(behaviour)
            if headroom is not None and headroom < 800:
                parts.append(
                    f"- PROMPT NEARLY FULL: prompt.md has only ~{headroom} "
                    "chars of headroom before the hard cap. To add strategy, "
                    "REPLACE/compress (write_harness_file) — appending will "
                    "be refused.")
            if parts:
                campaign_context = "## Campaign evidence (read carefully)\n"                     + "\n".join(parts)
        except Exception as e:
            print(f"[propose] WARNING: campaign context skipped ({e})")
        try:
            futile_sigs = set(futile)
        except NameError:
            futile_sigs = set()
        ctx[tid] = {
            "futile_sigs": futile_sigs,
            "base_spec": base_spec,
            "beam_entries": beam_entries,
            "beam_specs": beam_specs,
            "analysis": analysis_meta,
            "user_message": pio.build_user_message(
                task_id=tid, task_spec=task.spec,
                seed_program=seed_prog,
                seed_score=seed_sc,
                base_score=float(bases[tid]["score"]),
                base_spec=base_spec, max_evals=args.max_evals,
                campaign_context=campaign_context) + fb_text,
        }

    jobs = [(tid, k) for tid in args.tasks for k in range(args.k)]
    print(f"[propose] {len(args.tasks)} tasks x K={args.k} = {len(jobs)} H1 runs "
          f"across {len(base_urls)} replica(s)")

    # Structured exploration (opt-in via --force-tool-frac): a fraction of each
    # task's K candidates are told they MUST add a new tool. This breaks the
    # declarative-only prior of a phi trained before the generative genome
    # existed — like forcing an unexplored arm — without touching the fixed H1.
    force_frac = float(getattr(args, "force_tool_frac", 0.0) or 0.0)
    force_ks = set(range(min(args.k, max(0, round(args.k * force_frac)))))
    _FORCE_MSG = ("\n\n## REQUIRED FOR THIS CANDIDATE\nThis candidate MUST keep "
                  "at least one new mounted generated tool. A minimal starter "
                  "tool (quick_probe_k<slot>) is ALREADY written, mounted in "
                  "agent.yaml, and declared in prompt.md as your floor: it "
                  "passes validation as-is. Improve or replace its "
                  "implementation with a genuinely useful capability if you "
                  "can do so safely; otherwise keep it, make your other "
                  "improvements, then validate_harness and submit_harness. "
                  "Never delete your last generated tool.")

    # v2.6: exploration roles are ON by default.  With them off every slot
    # is an unconditioned "improve this harness" request and the measured
    # result is a batch of prompt edits; set SAH_H1_DIVERSITY=0 to restore
    # the unconditioned behaviour.
    diversity = os.environ.get("SAH_H1_DIVERSITY", "1") == "1"

    def _msg_for(tid, k):
        msg = ctx[tid]["user_message"]
        if k in force_ks:
            return msg + _FORCE_MSG
        return msg + pio.slot_role(k, force_ks, diversity, int(args.k))

    # REPAIR-001: the takeover agent must run on the FROZEN executor model,
    # never the trained proposer. Under split serving base_urls[0] is M_phi
    # and the last replica is frozen M0; without split serving every replica
    # is frozen, so the fallback is also safe.
    repair_enabled = os.environ.get("SAH_REPAIR", "0") == "1"
    repair_url = (
        os.environ.get("SAH_REPAIR_BASE_URL") or base_urls[-1]
    ) if repair_enabled else None
    repair_max_iters = int(os.environ.get("SAH_REPAIR_MAX_ITERS", "16"))

    def run(idx_job):
        idx, (tid, k) = idx_job
        seed = (args.seed * 1000 + idx) if args.seed is not None else None
        novelty_guard = os.environ.get("SAH_NOVELTY_GUARD", "0") == "1"
        # EXPLORE-001 (v2.5): the epsilon slot samples from the FROZEN BASE
        # replica so component-authoring behavior stays in the proposal (and
        # training) distribution; component-only slots are mechanically held
        # to an inherited prompt + one declaration line.
        k_total = int(args.k)
        comp_only = pio.component_only_slot(k, force_ks, diversity, k_total)
        role_name = pio.slot_role_name(k, force_ks, diversity, k_total)
        comp_kind = role_name if (comp_only or role_name == "middleware") else ""
        eps = pio.epsilon_slot(k, force_ks, diversity, k_total)
        # BEAM-001 dealing rule: the EXPLORATION slots (component-only and
        # epsilon) inherit the component lineage when one exists, so a
        # component can compound across rounds; every other slot keeps
        # improving the promoted package.  Width 1 => everyone on the head.
        beam = ctx[tid]["beam_entries"]
        b_idx = 1 if (len(beam) > 1 and (comp_only or eps)) else 0
        slot_base_spec = ctx[tid]["beam_specs"][b_idx]
        slot_url = (repair_url if (eps and repair_url)
                    else base_urls[idx % len(base_urls)])
        rec = pp.run_once(
            k, base_spec=slot_base_spec, user_message=_msg_for(tid, k),
            futile_signatures=(
                ctx[tid].get("futile_sigs") if novelty_guard else None),
            component_only=comp_only, component_kind=comp_kind,
            base_url=slot_url, model=args.model,
            api_key="EMPTY", seed=seed, timeout=args.llm_timeout,
            max_retries=args.llm_max_retries,
            require_new_tool=k in force_ks,
            repair_base_url=repair_url,
            repair_max_iters=repair_max_iters)
        rec.behavior_policy = (
            "frozen_base" if (eps and repair_url) else "trained_phi")
        rec.component_only = comp_only
        rec.parent_package = str(beam[b_idx]["package"])
        rec.parent_score = float(beam[b_idx]["score"])
        return tid, pp.enforce_new_tool_contract(
            rec, slot_base_spec, required=k in force_ks,
        )

    # -- optional sequential sampling (campaign_config: sampling.mode) -------- #
    # In parallel mode (default) the K samples of a task are independent. In
    # sequential mode, later samples see prior VALID actions from the same task
    # so they don't re-propose a paraphrase — Adaptive's within-batch diversity
    # policy. Tasks still run concurrently across the pool; only the K samples
    # WITHIN a task serialize. Zero change to the parallel path.
    seq = os.environ.get("SAH_SEQUENTIAL", "0") == "1"
    max_shared = int(os.environ.get("SAH_SEQ_MAX_SHARED", "6") or "6")

    def run_task_sequential(tid):
        recs, shared = [], []
        for k in range(args.k):
            seed = (args.seed * 1000 + k) if args.seed is not None else None
            msg = _msg_for(tid, k)
            if shared:
                msg = msg + pio.render_prior_actions(shared[-max_shared:])
            rec = pp.run_once(
                k, base_spec=ctx[tid]["base_spec"], user_message=msg,
                futile_signatures=(
                    ctx[tid].get("futile_sigs")
                    if os.environ.get("SAH_NOVELTY_GUARD", "0") == "1"
                    else None),
                base_url=base_urls[k % len(base_urls)], model=args.model,
                api_key="EMPTY", seed=seed, timeout=args.llm_timeout,
                max_retries=args.llm_max_retries,
                repair_base_url=repair_url,
                repair_max_iters=repair_max_iters,
                require_new_tool=k in force_ks)
            pp.enforce_new_tool_contract(
                rec, ctx[tid]["base_spec"], required=k in force_ks,
            )
            recs.append((tid, rec))
            if rec.valid and rec.effective is not None:
                shared.append({"k": k, "changed_fields": rec.changed_fields,
                               "spec": rec.spec})
        return recs

    if seq:
        print(f"[propose] sequential sampling ON (share <= {max_shared} prior valid actions)")
        with ThreadPoolExecutor(max_workers=min(args.parallel, len(args.tasks))) as ex:
            results = [r for task_recs in ex.map(run_task_sequential, args.tasks)
                       for r in task_recs]
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            results = list(ex.map(run, enumerate(jobs)))

    per_task: Dict[str, Any] = {}
    trajectories = []
    repair_trajectories = []
    for tid in args.tasks:
        records = sorted((r for t, r in results if t == tid), key=lambda r: r.k)
        pp.dedup_group(records, ctx[tid]["base_spec"])
        cands = []
        for rec in records:
            entry = {"k": rec.k, "valid": rec.valid, "errors": rec.errors,
                     "behavior_policy": getattr(rec, "behavior_policy", None),
                     "component_only": getattr(rec, "component_only", False),
                     "parent_package": getattr(rec, "parent_package", None),
                     "parent_score": getattr(rec, "parent_score", None),
                     # BEAM-001: the paired contract must be checked against
                     # THIS slot's parent, not the task's head package.
                     "parent_package_sha256": (
                         h2_sha256(Path(rec.parent_package))
                         if getattr(rec, "parent_package", None) else None),
                     "spec_hash": rec.spec_hash, "changed_fields": rec.changed_fields,
                     "stop_reason": rec.stop_reason, "llm_calls": rec.llm_calls,
                     "tool_schema_guard": getattr(rec, "schema_guard", {}),
                     "repair": getattr(rec, "repair", {}),
                     "review_log": getattr(rec, "review_log", []),
                     "force_tool_required": rec.k in force_ks,
                     "force_tool_compliant": (
                         bool(hs.added_generated_component_names(
                             ctx[tid]["base_spec"], rec.effective or {},
                             "new_tools",
                         ))
                         if rec.k in force_ks else None
                     )}
            if rec.valid:
                cdir = round_dir / "tasks" / tid / f"cand{rec.k:02d}"
                component_lineage = hs.generated_component_lineage(
                    ctx[tid]["base_spec"], rec.effective
                )
                materialize(rec.effective, cdir, raw_spec_text=rec.raw_submission,
                            meta={"round": args.round, "task_id": tid, "k": rec.k,
                                  "spec_hash": rec.spec_hash,
                                  "changed_fields": rec.changed_fields,
                                  "base_package": bases[tid]["package"],
                                  "base_component_inventory":
                                      hs.generated_component_inventory(
                                          ctx[tid]["base_spec"]),
                                  "component_lineage": component_lineage,
                                  "effective": rec.effective})
                entry["dir"] = str(cdir)
                entry["package_sha256"] = h2_sha256(cdir)
                entry["component_lineage"] = component_lineage
            cands.append(entry)
            trajectories.append({"task_id": tid, "k": rec.k,
                                 "raw_submission": rec.raw_submission,
                                 "trajectory": rec.trajectory})
            if getattr(rec, "repair_trajectory", None):
                repair_trajectories.append({
                    "task_id": tid, "k": rec.k,
                    "repair": getattr(rec, "repair", {}),
                    "trajectory": rec.repair_trajectory,
                })
        n_ok = sum(c["valid"] for c in cands)
        print(f"  {tid}: {n_ok}/{len(cands)} valid | " + " ".join(
            f"c{c['k']:02d}{'+' if c['valid'] else '-'}" for c in cands))
        per_task[tid] = {"base_package": bases[tid]["package"],
                         "base_package_sha256": h2_sha256(
                             Path(bases[tid]["package"])
                         ),
                         "base_score": bases[tid]["score"],
                         "seed_score": bases[tid]["seed_score"],
                         "base_spec_hash": hs.spec_hash(ctx[tid]["base_spec"]),
                         "analysis": ctx[tid]["analysis"],
                         "candidates": cands}

    # Fail-closed proposal gate (SAH_PROPOSAL_GATE=all): a causal comparison
    # round must not proceed to rollouts unless every charged slot holds a
    # valid materialized candidate. The audit file is written on both outcomes
    # so a passing round proves the gate actually ran; on violation the
    # proposer evidence (prompts + trajectories + audit) is preserved and the
    # round fails before round.json exists, so a retry re-proposes cleanly.
    gate_mode = os.environ.get("SAH_PROPOSAL_GATE", "off")
    try:
        gate_kind, gate_min_valid = pp.parse_gate_mode(gate_mode, args.k)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    gate_tasks: Dict[str, Any] = {}
    for tid in args.tasks:
        cands = per_task[tid]["candidates"]
        slot_issues = (
            pp.proposal_gate_issues(cands, k=args.k)
            if gate_kind in ("all", "min") else []
        )
        valid_slots = sum(bool(c.get("valid")) for c in cands)
        if gate_kind == "all":
            issues = slot_issues
        elif gate_kind == "min" and valid_slots < gate_min_valid:
            issues = slot_issues + [
                f"only {valid_slots}/{len(cands)} valid slots "
                f"(minimum {gate_min_valid} required)"
            ]
        else:
            issues = []
        gate_tasks[tid] = {
            "charged_slots": len(cands),
            "valid_slots": valid_slots,
            "repaired_slots": sum(
                1 for c in cands
                if (c.get("repair") or {}).get("succeeded")
            ),
            "min_valid_required": gate_min_valid,
            "slot_issues": slot_issues,
            "issues": issues,
        }
    gate_passed = all(not row["issues"] for row in gate_tasks.values())
    (round_dir / "proposal_gate_audit.json").write_text(json.dumps({
        "schema": "proposal-gate/1.0",
        "mode": gate_mode,
        "enforced": gate_kind in ("all", "min"),
        "k": int(args.k),
        "passed": gate_passed,
        "tasks": gate_tasks,
    }, indent=2))
    if not gate_passed:
        (round_dir / "prompts.json").write_text(json.dumps(
            {tid: ctx[tid]["user_message"] for tid in args.tasks}, indent=2))
        (round_dir / "trajectories.json").write_text(
            json.dumps(trajectories, indent=2))
        if repair_trajectories:
            (round_dir / "repair_trajectories.json").write_text(
                json.dumps(repair_trajectories, indent=2))
        # A requeued attempt re-runs propose and overwrites the round-local
        # files — archive each failed attempt's evidence immutably (failures
        # must be retained and remain diagnosable).
        archive_root = round_dir / "proposal_gate_failures"
        archive_root.mkdir(exist_ok=True)
        attempt = 1 + sum(1 for p in archive_root.iterdir() if p.is_dir())
        archive = archive_root / f"attempt_{attempt:02d}"
        archive.mkdir()
        for name in ("proposal_gate_audit.json", "prompts.json",
                     "trajectories.json"):
            (archive / name).write_bytes((round_dir / name).read_bytes())
        detail = "; ".join(
            f"{tid} {row['valid_slots']}/{row['charged_slots']} valid"
            for tid, row in gate_tasks.items() if row["issues"]
        )
        raise RuntimeError(
            f"proposal gate failed (SAH_PROPOSAL_GATE={gate_mode}): refusing to charge "
            f"intervention slots for invalid H2 proposals — {detail}; "
            f"evidence archived in {archive}"
        )

    fixed_inference_slots = (
        os.environ.get("SAH_FIXED_INFERENCE_SLOTS", "0") == "1"
    )
    paired_controls = (
        os.environ.get("SAH_PAIRED_CONTROLS", "0") == "1"
        or os.environ.get("SAH_ADV", "") == "paired"
    )
    paired_repeats = (
        int(os.environ.get("SAH_PAIRED_REPEATS", "1"))
        if paired_controls else 1
    )
    if paired_repeats < 1:
        raise RuntimeError("SAH_PAIRED_REPEATS must be >= 1")
    logical_round_raw = os.environ.get("LOGICAL_ROUND_INDEX", "")
    logical_round_index = (
        int(logical_round_raw) if logical_round_raw.strip() else None
    )
    if fixed_inference_slots and logical_round_index is None:
        raise RuntimeError(
            "fixed inference slots require LOGICAL_ROUND_INDEX for x-axis provenance"
        )
    decode_round_index = logical_round_index or 0
    h2_slots = int(args.k) * (paired_repeats if paired_controls else 1)
    control_slots = int(args.k) * paired_repeats if paired_controls else 0
    nominal_total = (
        int(args.k) + h2_slots + control_slots
        if fixed_inference_slots else None
    )
    h2_seed_stride = max(16, int(args.k) * paired_repeats)
    shared_anchor_trajectories = int(
        os.environ.get("SAH_SHARED_ANCHOR_TRAJECTORIES", "1")
    )
    if shared_anchor_trajectories < 0:
        raise RuntimeError("SAH_SHARED_ANCHOR_TRAJECTORIES must be >= 0")
    (round_dir / "round.json").write_text(json.dumps({
        "round": args.round, "created": time.strftime("%Y%m%d-%H%M%S"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "mode": "instance_wise",
        "h1_version": pio.H1_VERSION, "h1_package_hash": pio.h1_hash(),
        "proposer": {
            "base_urls": base_urls,
            "model": args.model,
            "seed": args.seed,
            "checkpoint": (
                os.environ.get("MPHI_PATH")
                or os.environ.get("MODEL_PATH")
                or None
            ),
            "executor_checkpoint": os.environ.get("MODEL_PATH") or None,
        },
        "tasks_order": args.tasks, "max_evals": args.max_evals, "k": args.k,
        "proposal_contract": {
            "force_tool_fraction": force_frac,
            "force_tool_slots": sorted(force_ks),
            "enforcement": "new-mounted-generated-tool-postcondition",
        },
        "cross_round_inputs": {
            "bases": bases_input,
            "feedback": feedback_input,
            "seed_programs": program_input,
        },
        "inference_trajectory_budget": {
            "schema": 1,
            "axis_unit": "generated_agent_trajectory",
            "fixed_h1_plus_h2_slots": fixed_inference_slots,
            "logical_round_index": logical_round_index,
            "h1_slots_per_task": int(args.k),
            "h2_slots_per_task": (
                h2_slots if fixed_inference_slots else None
            ),
            "paired_control_slots_per_task": (
                control_slots if fixed_inference_slots and paired_controls else 0
            ),
            "nominal_total_slots_per_task": nominal_total,
            "shared_anchor_x": (
                shared_anchor_trajectories if fixed_inference_slots else None
            ),
            "axis_x_after_round": (
                shared_anchor_trajectories
                + (logical_round_index + 1) * nominal_total
                if fixed_inference_slots else None
            ),
            "h2_decode_seed_base": int(os.environ.get("H2_SEED_BASE", "200000")),
            "h2_decode_seed_stride": h2_seed_stride,
            "h2_decode_seeds": [
                int(os.environ.get("H2_SEED_BASE", "200000"))
                + decode_round_index * h2_seed_stride + k
                for k in range(int(args.k))
            ],
            "h2_decode_seed_matrix": [
                [
                    int(os.environ.get("H2_SEED_BASE", "200000"))
                    + decode_round_index * h2_seed_stride
                    + repeat * int(args.k) + k
                    for repeat in range(paired_repeats)
                ]
                for k in range(int(args.k))
            ],
            "analyzer_calls_excluded_from_axis": True,
            "post_submit_reviewer_model_calls": 0,
        },
        "causal_attribution": {
            "schema": "paired-parent-control/1.0",
            "enabled": paired_controls,
            "candidate_root": "rollouts",
            "control_root": "paired_controls" if paired_controls else None,
            "pairing": "same_task_program_budget_model_decode_seed",
            "repeats_per_candidate": paired_repeats if paired_controls else 0,
        },
        "program_ratchet_mode": os.environ.get(
            "SAH_PROGRAM_RATCHET_MODE", "legacy_qd"
        ),
        "eval_timeout_s": int(float(os.environ.get("EVAL_TIMEOUT", "0") or 0)),
        "trajectory_wall_timeout_s": int(float(
            os.environ.get("ROLLOUT_WALL_TIMEOUT", "0") or 0
        )),
        "proposal_parallelism": {
            "requested_workers": int(args.parallel),
            "serving_replicas": len(base_urls),
        },
        "bases_in": bases,
        "per_task": per_task,
    }, indent=2))
    (round_dir / "prompts.json").write_text(json.dumps(
        {tid: ctx[tid]["user_message"] for tid in args.tasks}, indent=2))
    (round_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2))
    if repair_trajectories:
        # Separate file: trajectories.json stays exactly one primary (phi)
        # trajectory per slot for the training/audit contracts.
        (round_dir / "repair_trajectories.json").write_text(
            json.dumps(repair_trajectories, indent=2))
    total_ok = sum(c["valid"] for t in per_task.values() for c in t["candidates"])
    n_rep = sum(
        1 for t in per_task.values() for c in t["candidates"]
        if (c.get("repair") or {}).get("succeeded")
    )
    print(f"[propose] {total_ok}/{len(jobs)} valid candidates "
          f"({n_rep} repaired by frozen M0) -> {round_dir}")


def cmd_collect(args) -> None:
    round_dir = Path(args.round_dir)
    meta = json.loads((round_dir / "round.json").read_text())
    prompts = json.loads((round_dir / "prompts.json").read_text())
    trajs = {(t["task_id"], t["k"]): t
             for t in json.loads((round_dir / "trajectories.json").read_text())}
    system_text = (pio.H1_PACKAGE / "system.md").read_text()

    adv_mode = os.environ.get("SAH_ADV", "v2")
    # v3 == v2 except on no_signal groups, so results alone can't tell you which
    # impl ran — log it so collect.log is proof the env survived sbatch/srun.
    print(f"[collect] advantage impl: SAH_ADV={adv_mode} "
          f"(hist_lambda={os.environ.get('SAH_HIST_LAMBDA', '-')})")
    ceilings = {}
    try:
        ft = json.loads((Path(__file__).resolve().parents[3]
                         / "results" / "finch_targets.json").read_text())
        ceilings = {t: (v.get("sota_combined") or v.get("finch_combined"))
                    for t, v in ft.get("targets", {}).items()}
    except Exception:
        pass

    groups, batch_rows = {}, []
    next_bases = dict(meta.get("bases_in", {}))  # carry forward ALL tasks' bases
    for tid in meta["tasks_order"]:
        pt = meta["per_task"][tid]
        if adv_mode == "legacy":
            g = rw.compute_task_group(
                task_id=tid, candidates=pt["candidates"],
                rollout_root=round_dir / "rollouts", base_score=float(pt["base_score"]))
        elif adv_mode == "paired":
            attribution = meta.get("causal_attribution") or {}
            if attribution.get("enabled") is not True:
                raise RuntimeError(
                    "SAH_ADV=paired requires paired controls declared at "
                    "proposal time (SAH_PAIRED_CONTROLS=1)"
                )
            g = rw.compute_task_group_paired(
                task_id=tid, candidates=pt["candidates"],
                rollout_root=round_dir / "rollouts",
                control_root=round_dir / "paired_controls",
                base_score=float(pt["base_score"]),
                ceiling=ceilings.get(tid),
                sharpen_alpha=float(os.environ.get("SAH_ALPHA", "0.3")),
                min_effect=float(os.environ.get("SAH_MIN_PAIRED_EFFECT", "0")),
                expected_control_h2_sha256=pt.get("base_package_sha256"),
                expected_repeats=int(attribution.get("repeats_per_candidate", 1)),
            )
            print(
                f"  [{tid}] advantages: {g['adv_mode']} "
                f"paired_delta={g.get('best_causal_delta')}"
            )
        elif adv_mode == "anchored":
            g = rw.compute_task_group_anchored(
                task_id=tid, candidates=pt["candidates"],
                rollout_root=round_dir / "rollouts", base_score=float(pt["base_score"]),
                ceiling=ceilings.get(tid))
            print(f"  [{tid}] advantages: {g['adv_mode']} ceiling={g.get('ceiling')}")
        elif adv_mode == "v3":
            g = rw.compute_task_group_v3(
                task_id=tid, candidates=pt["candidates"],
                rollout_root=round_dir / "rollouts", base_score=float(pt["base_score"]),
                ceiling=ceilings.get(tid), sharpen_alpha=float(os.environ.get("SAH_ALPHA", "0.3")),
                hist_lambda=float(os.environ.get("SAH_HIST_LAMBDA", "0.3")))
            print(f"  [{tid}] advantages: {g['adv_mode']} ceiling={g.get('ceiling')}")
        else:
            g = rw.compute_task_group_v2(
                task_id=tid, candidates=pt["candidates"],
                rollout_root=round_dir / "rollouts", base_score=float(pt["base_score"]),
                ceiling=ceilings.get(tid), sharpen_alpha=float(os.environ.get("SAH_ALPHA", "0.3")))
            print(f"  [{tid}] advantages: {g['adv_mode']} ceiling={g.get('ceiling')}")
        groups[tid] = g
        for row in g["rows"]:
            t = trajs.get((tid, row["k"]), {})
            batch_rows.append({
                "round": meta["round"], "task_id": tid, "k": row["k"],
                "system": system_text, "user": prompts[tid],
                "response": t.get("raw_submission", ""),
                "trajectory": t.get("trajectory", []),
                "reward": row["reward"], "advantage": row["advantage"],
                "valid": row["valid"], "score": row["score"],
                "spec_hash": row["spec_hash"],
                "control_score": row.get("control_score"),
                "causal_delta": row.get("causal_delta"),
                "attribution_valid": row.get("attribution_valid"),
                "attribution_status": row.get("attribution_status"),
                "pair_contract": row.get("pair_contract"),
            })
        # next round's base for this task = best candidate iff it beat the base
        if g["improved"]:
            next_bases[tid] = {
                "package": str(round_dir / "tasks" / tid / f"cand{g['best_k']:02d}"),
                "score": g["best_score"], "seed_score": pt["seed_score"],
                "from": f"round{meta['round']:03d}/cand{g['best_k']:02d}"}
        else:
            # BASEREFRESH-001: the paired controls ARE rollouts of the
            # retained incumbent; once enough of them agree, they are a
            # better estimate of its score than the 2-seed measurement from
            # its promotion round (observed: base recorded 0.9393 while 112
            # subsequent same-harness rollouts scored 0.9645).  Refresh with
            # full provenance; the ratchet inherits the honest number.
            ctrl_scores = [
                float(row["control_score"]) for row in g.get("rows", [])
                if row.get("control_score") is not None
            ]
            refreshed = pt["base_score"]
            refresh_note = None
            if len(ctrl_scores) >= 4:
                est = sum(ctrl_scores) / len(ctrl_scores)
                if abs(est - pt["base_score"]) > 1e-3:
                    refreshed = est
                    refresh_note = {
                        "refreshed_from": "paired_control_rollouts",
                        "samples": len(ctrl_scores),
                        "previous_score": pt["base_score"],
                    }
            next_bases[tid] = {"package": pt["base_package"],
                               "score": refreshed,
                               "seed_score": pt["seed_score"], "from": "unchanged"}
            if refresh_note:
                next_bases[tid]["score_refresh"] = refresh_note
        # BEAM-001: publish the inheritance beam alongside the incumbent.
        # beam[0] is always the promoted/retained package (the curve score);
        # beam[1..] are exploration parents, chosen to PRESERVE DIVERSITY:
        # the best component-carrying candidate within BEAM_TOL of the best
        # wins the slot, else the runner-up by score.  Width 1 reproduces
        # strict single inheritance byte-for-byte.
        beam_width = int(os.environ.get("CURVE_INHERIT_BEAM", "1") or 1)
        if beam_width > 1:
            tol = float(os.environ.get("CURVE_BEAM_TOL", "0.02") or 0.02)
            head = {"package": next_bases[tid]["package"],
                    "score": next_bases[tid]["score"], "from": "incumbent"}
            pool = []
            for row in g.get("rows", []):
                if row.get("score") is None or not row.get("valid", True):
                    continue
                cand_dir = str(round_dir / "tasks" / tid
                               / f"cand{int(row['k']):02d}")
                if cand_dir == head["package"]:
                    continue
                has_component = bool([
                    f for f in (row.get("changed_fields") or [])
                    if not str(f).startswith("system_prompt")
                    and not str(f).startswith("new_tools.quick_probe")
                ])
                pool.append({"package": cand_dir, "score": float(row["score"]),
                             "from": f"round{meta['round']:03d}/cand{int(row['k']):02d}",
                             "has_component": has_component})
            pool.sort(key=lambda e: -e["score"])
            beam = [head]
            near = [e for e in pool
                    if e["has_component"] and head["score"] - e["score"] <= tol]
            if near:
                beam.append(near[0])
            for entry in pool:
                if len(beam) >= beam_width:
                    break
                if entry["package"] not in {b["package"] for b in beam}:
                    beam.append(entry)
            next_bases[tid]["beam"] = beam[:beam_width]
            next_bases[tid]["beam_policy"] = {
                "width": beam_width, "tolerance": tol,
                "rule": "head=promoted/retained; then best component-carrying "
                        "candidate within tolerance; then by score",
            }
        bs = f"{g['best_score']:.6g}" if g["best_score"] is not None else "n/a"
        print(f"  {tid}: base={g['base_score']:.6g} best={bs} "
              f"(cand{g['best_k']:02d})" if g["best_k"] is not None else
              f"  {tid}: base={g['base_score']:.6g} best=n/a",
              "IMPROVED" if g["improved"] else "")

    # inner-telemetry digest for the NEXT visit's H1 context (blind mutation ->
    # informed debugging): what each candidate's rollout actually did
    fb_path = round_dir.parent / "task_feedback.json"
    try:
        feedback = json.loads(fb_path.read_text()) if fb_path.exists() else {}
    except Exception:
        feedback = {}
    for tid, g in groups.items():
        cands = []
        for row in g["rows"]:
            ent = {"k": row["k"], "score": row["score"],
                   "control_score": row.get("control_score"),
                   "causal_delta": row.get("causal_delta"),
                   "attribution_status": row.get("attribution_status"),
                   "changed": [c.split(".")[-1] for c in row.get("changed_fields", [])][:6]}
            rl = row.get("review_log") or []
            if rl:
                ent["tools"] = [f"{x['name']}:{'ok' if x['ok'] else 'dropped('+str(x.get('error',''))[:40]+')'}" for x in rl]
            for res in sorted((round_dir / "rollouts" / tid / f"cand{row['k']:02d}").glob("*/results/*.json")):
                try:
                    d = json.loads(res.read_text())
                    led = d.get("ledger") or {}
                    ent.update({"evals": led.get("evaluator_calls"),
                                "llm_calls": led.get("llm_calls"),
                                "stop": d.get("stop_reason"),
                                "err": (str(d.get("error"))[:120] if d.get("error") else None)})
                except Exception:
                    pass
            if not row["valid"]:
                ent["invalid"] = True
            cands.append(ent)
        n_stuck = sum(1 for c in cands if c.get("score") is not None
                      and abs((c["score"] or 0) - g["base_score"]) < 1e-9)
        prev_note = (feedback.get(tid) or {}).get("analyst_note")
        feedback[tid] = {"round": meta["round"], "base_score": g["base_score"],
                         "best_score": g["best_score"], "best_k": g["best_k"],
                         "n_stuck_at_base": n_stuck, "candidates": cands}
        if prev_note:  # analyst notes are curated externally — never wipe them
            # anti-leak (campaign_config: leakage_guard): the analyst_note is the
            # only free-form text reaching the proposer; drop it if it carries a
            # leak marker, else neutralize unproven result-claim words so they
            # can't survive as facts. Default-on; a no-op on clean structured notes.
            if os.environ.get("SAH_LEAK_NEUTRALIZE", "1") == "1":
                from outer.safety.leak_guard import code_signals, sanitize_note
                dropped_for = code_signals(prev_note)
                prev_note = sanitize_note(prev_note)
                if dropped_for:
                    print(f"[leak-guard] {tid}: analyst_note DROPPED "
                          f"(code-injection signals: {dropped_for})")
            if prev_note:
                feedback[tid]["analyst_note"] = prev_note
    fb_path.write_text(json.dumps(feedback, indent=1))
    (round_dir / "task_feedback_after.json").write_text(
        json.dumps(feedback, indent=1)
    )

    # Route-independent program inheritance.  Canonical comparison runs set
    # SAH_PROGRAM_RATCHET_MODE=strict_single; legacy exploratory campaigns keep
    # their historical QD/crossover memory behind an explicit mode.
    bp_path = round_dir.parent / "best_programs.json"
    ratchet_mode = os.environ.get("SAH_PROGRAM_RATCHET_MODE", "legacy_qd")
    ratchet_mode_source = (
        "env" if "SAH_PROGRAM_RATCHET_MODE" in os.environ else "default"
    )
    if (os.environ.get("SAH_REQUIRE_STRICT_RATCHET", "0") == "1"
            and ratchet_mode != "strict_single"):
        raise RuntimeError(
            "SAH_REQUIRE_STRICT_RATCHET=1 but program ratchet mode is "
            f"{ratchet_mode!r} ({ratchet_mode_source}); canonical campaigns "
            "must run strict_single"
        )
    logical_index = (
        meta.get("inference_trajectory_budget") or {}
    ).get("logical_round_index")
    try:
        best_programs = json.loads(bp_path.read_text()) if bp_path.exists() else {}
        if not isinstance(best_programs, dict):
            raise TypeError("program incumbent registry must be a JSON object")
    except Exception as exc:
        if ratchet_mode == "strict_single":
            raise RuntimeError(
                f"canonical program incumbent registry is unreadable: {bp_path}"
            ) from exc
        best_programs = {}
    if ratchet_mode == "strict_single" and logical_index not in (None, 0) \
            and not bp_path.is_file():
        raise RuntimeError(
            f"canonical post-cold round lacks program incumbent registry: {bp_path}"
        )
    best_programs, ratchet_audit = update_program_ratchet(
        mode=ratchet_mode,
        round_dir=round_dir,
        groups=groups,
        bases_in=meta.get("bases_in", {}),
        previous=best_programs,
        round_id=int(meta["round"]),
    )
    for tid, row in ratchet_audit["tasks"].items():
        groups[tid]["program_ratchet"] = row
        groups[tid]["accepted_improvement"] = bool(row.get("promoted"))
        if ratchet_mode == "strict_single" and groups[tid].get("improved") \
                and not row.get("promoted"):
            # A raw score cannot advance H2 when it cannot be bound to a newly
            # improved program from the same rollout.  This closes the legacy
            # step-0/inherited-program attribution hole.
            pt = meta["per_task"][tid]
            next_bases[tid] = {
                "package": pt["base_package"],
                "score": pt["base_score"],
                "seed_score": pt["seed_score"],
                "from": "unchanged_program_attribution_failed",
            }
            # Do not let an unbound step-0/noisy score update proposer weights
            # indirectly.  Keeping the diagnostic rewards while zeroing every
            # task-row advantage makes replay conversion skip this update.
            suppress_task_training(batch_rows, tid, row.get("reason"))
            groups[tid]["training_suppressed"] = True
            groups[tid]["training_suppression_reason"] = row.get("reason")
        # COMPRATCHET-001 runs HERE, not in the per-task loop above: the
        # strict_single revert replaces next_bases[tid] wholesale, which
        # silently discarded a graft the ledger had already recorded as
        # admitted (and left the graft dir orphaned on disk).  Grafting after
        # the base is final also means components land on the package that
        # actually gets inherited.
        try:
            from outer.rounds.component_ratchet import apply_component_ratchet
            ratchet = apply_component_ratchet(
                round_dir=round_dir, tid=tid, round_index=int(meta["round"]),
                candidates=(meta["per_task"][tid].get("candidates") or []),
                rows=groups[tid].get("rows") or [],
                improved=bool(groups[tid].get("improved")),
                best_k=groups[tid].get("best_k"),
                next_base_entry=next_bases[tid],
            )
            if ratchet and ratchet.get("admitted"):
                names = ", ".join(a["name"] for a in ratchet["admitted"])
                print(f"  [{tid}] component ratchet admitted: {names} "
                      f"-> {next_bases[tid]['package']}")
        except Exception as exc:
            print(f"  [{tid}] component ratchet skipped on error: {exc}")
        if tid in feedback:
            feedback[tid]["accepted_improvement"] = bool(row.get("promoted"))
            feedback[tid]["program_ratchet_reason"] = row.get("reason")
            feedback[tid]["outgoing_base_score"] = next_bases[tid]["score"]
        print(
            f"  [program-ratchet:{ratchet_mode}] {tid}: "
            f"promoted={row.get('promoted')} reason={row.get('reason', 'legacy')}"
        )

    bp_path.write_text(json.dumps(best_programs, indent=1))
    (round_dir / "best_programs_after.json").write_text(
        json.dumps(best_programs, indent=1)
    )
    ratchet_audit["mode_source"] = ratchet_mode_source
    ratchet_audit["strict_required"] = (
        os.environ.get("SAH_REQUIRE_STRICT_RATCHET", "0") == "1"
    )
    (round_dir / "program_ratchet_audit.json").write_text(
        json.dumps(ratchet_audit, indent=2)
    )
    # Rewrite after the joint H2/program transition so the next context sees
    # the accepted state, not merely a noisy/raw candidate maximum.
    fb_path.write_text(json.dumps(feedback, indent=1))
    (round_dir / "task_feedback_after.json").write_text(
        json.dumps(feedback, indent=1)
    )

    with open(round_dir / "grpo_batch.jsonl", "w") as f:
        for row in batch_rows:
            f.write(json.dumps(row) + "\n")
    # Publish every transition dependency before the summary marker.  Resumable
    # campaign workers treat round_summary.json as the commit record and must
    # never observe it without the corresponding next_bases registry.
    (round_dir / "next_bases.json").write_text(json.dumps(next_bases, indent=2))
    (round_dir / "round_summary.json").write_text(json.dumps({
        "round": meta["round"], "groups": groups,
        "inference_trajectory_budget": meta.get("inference_trajectory_budget"),
        "improved_tasks": [t for t, g in groups.items() if g["improved"]],
        "accepted_improved_tasks": [
            t for t, g in groups.items() if g.get("accepted_improvement")
        ],
    }, indent=2))
    n_imp = sum(bool(g.get("accepted_improvement")) for g in groups.values())
    print(f"[collect] {len(batch_rows)} GRPO rows | {n_imp}/{len(groups)} tasks accepted-improved "
          f"-> grpo_batch.jsonl, round_summary.json, next_bases.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose", help="K H1 runs per task; validate; materialize")
    p.add_argument("--round-dir", required=True)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--bases-file", default=None,
                   help="per-task bases registry (prev round's next_bases.json)")
    p.add_argument("--base-url", default="http://127.0.0.1:8800/v1")
    p.add_argument("--n-replicas", type=int, default=0,
                   help="if >0, fan H1 runs from --base-url across consecutive ports")
    p.add_argument("--model", default="qwen3.5-9b")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--max-evals", type=int, default=20)
    p.add_argument("--llm-timeout", type=float, default=600.0,
                   help="per-request timeout for each proposer trajectory")
    p.add_argument("--llm-max-retries", type=int, default=2,
                   help="transport retries for each proposer request")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    p.add_argument("--force-tool-frac", type=float, default=0.0,
                   help="fraction of each task's K candidates required to add a new tool (0..1)")
    p.add_argument("--seed-programs-file", default=None,
                   help="global best_programs.json; H1 sees the inherited program as the rollout starting point")
    p.add_argument("--feedback-file", default=None,
                   help="global task_feedback.json; H1 sees a telemetry digest of the previous visit's rollouts")
    p.set_defaults(fn=cmd_propose)

    c = sub.add_parser("collect", help="per-task rewards + GRPO batch + next bases")
    c.add_argument("--round-dir", required=True)
    c.set_defaults(fn=cmd_collect)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
