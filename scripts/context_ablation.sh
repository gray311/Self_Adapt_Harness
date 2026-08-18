#!/usr/bin/env bash
# \method (context) ablation — adaptation through the proposer's CONTEXT only.
#
#   context_ablation.sh <n_steps> <round_base> [workspace]
#
# The proposer's weights are NEVER updated (no train step at all).  What changes
# round to round is only what the proposer READS: the incumbent harness/program,
# the experience digest, and an analyst brief produced by the FROZEN executor.
# This isolates context adaptation from weight adaptation, and is the row that
# separates "harness search helps" from "training the proposer helps".
#
# Leakage discipline (the whole point of this run):
#   * proposer = frozen base = same weights as the executor (no stronger model)
#   * analyst  = the SAME frozen model, via SAH_ANALYSIS_BASE_URL
#   * SAH_LEAK_NEUTRALIZE=1 scrubs the brief before it reaches the proposer
#   * no curated notes: any analyst_note in the feedback file is stripped below
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"
OUT_TAG="${OUT_TAG:-}"
OUT="$RUN_ROOT/self_adapt_harness/outer${OUT_TAG:+-$OUT_TAG}"
NSTEPS="${1:?usage: context_ablation.sh <n_steps> <round_base> [ws]}"
RBASE="${2:?}"
WS="${3:-$RUN_ROOT/self_adapt_harness/context_ablation}"
mkdir -p "$WS"
log(){ echo "[$(date -Is)] [ctx] $*"; }

# AHC tasks score 0 unless the natively-rebuilt aarch64 testers are used (the
# stock x86 binaries fail silently under qemu). Harmless for the other tasks.
export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS=12 AHC_CACHE_DIR="$SAH/ahc_work/cache"

TASKS_ALL="${CTX_TASKS:-eft__math__erdos_min_overlap eft__math__first_autocorr_ineq eft__math__second_autocorr_ineq eft__math__circle_packing eft__math__hadamard_maximal_det eft__ahc_simpletes__ahc039 eft__ahc_simpletes__ahc058 adrs__eplb adrs__prism adrs__llm_sql adrs__txn_scheduling}"

# Record the exact output namespace and knobs.  A non-empty OUT_TAG is required
# for new paper ablations: collect writes feedback next to the round directory,
# so a private outer-* root prevents concurrent campaigns from reading one
# another's telemetry.
python3 - "$WS/run_manifest.json" "$OUT" "$OUT_TAG" "$TASKS_ALL" "$RBASE" \
  "$NSTEPS" "${K:-8}" "${MAX_EVALS:-20}" "${RESUME_BASES:-}" <<'PY'
import json, sys, time
path, outer_root, out_tag, tasks, round_base, nsteps, k, max_evals, resume = sys.argv[1:]
try:
    payload = json.load(open(path))
except Exception:
    payload = {
        "schema": 2,
        "method": "context-only analyzer; frozen proposer and executor weights",
        "segments": [],
    }
payload.update({
    "schema": 1,
    "method": "context-only analyzer; frozen proposer and executor weights",
    "outer_root": outer_root,
    "out_tag": out_tag,
    "isolated_feedback": bool(out_tag),
    "tasks": tasks.split(),
    "round_base": int(round_base),
    "k_per_round": int(k),
    "max_evals_per_trajectory": int(max_evals),
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
})
payload["schema"] = 2
payload.setdefault("segments", []).append({
    "round_base": int(round_base),
    "n_steps": int(nsteps),
    "resume_bases": resume or None,
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
})
with open(path, "w") as f:
    json.dump(payload, f, indent=2)
PY
log "round root: $OUT (isolated_feedback=$([ -n "$OUT_TAG" ] && echo yes || echo no))"

# seed the harness base from the fixed initial harness (same start as the other rows)
bases="$WS/round000_bases.json"
if [ ! -f "$bases" ]; then
  python3 - "$bases" "$SAH" <<'PY'
import json,sys
out,sah=sys.argv[1],sys.argv[2]
tasks="eft__math__erdos_min_overlap eft__math__first_autocorr_ineq eft__math__second_autocorr_ineq eft__math__circle_packing eft__math__hadamard_maximal_det eft__ahc_simpletes__ahc039 eft__ahc_simpletes__ahc058 adrs__eplb adrs__prism adrs__llm_sql adrs__txn_scheduling".split()
baseline=json.load(open(f"{sah}/results/baseline_h2_20ev.json"))["baseline"]
json.dump({t:{"package":f"{sah}/src/inner/harness",
              "score":float(baseline[t]["h2_best"]),
              "seed_score":float(baseline[t]["seed"])}
           for t in tasks}, open(out,"w"), indent=1)
print("seeded initial-harness bases for", len(tasks), "tasks")
PY
fi
if [ -n "${RESUME_BASES:-}" ]; then
  [ -s "$RESUME_BASES" ] || { log "RESUME_BASES missing: $RESUME_BASES"; exit 2; }
  bases="$RESUME_BASES"
  log "resuming bases from $bases"
fi

wait_job(){
  local J="$1"
  while :; do
    local S; S=$(timeout 20s squeue -j "$J" -h -o %T 2>/dev/null | head -1 || true)
    case "$S" in PENDING|RUNNING|COMPLETING) sleep 60; continue ;; esac
    local ST; ST=$(timeout 20s sacct -j "$J" -X -n -o State 2>/dev/null | head -1 | xargs || true)
    case "$ST" in PENDING|RUNNING|COMPLETING|"") sleep 60 ;; *) return 0 ;; esac
  done
}

for i in $(seq 0 $((NSTEPS-1))); do
  [ -f "$WS/STOP" ] && { log "STOP flag"; break; }
  if [ -n "${RUNTIME_SOURCE_MANIFEST:-}" ]; then
    python3 "$SAH/scripts/runtime/provenance.py" verify \
      --manifest "$RUNTIME_SOURCE_MANIFEST" || exit 43
  fi
  R=$((RBASE+i)); RD="$OUT/round$(printf '%03d' "$R")"; prev_bases="$bases"
  LOGICAL_PROPOSER_SEED=$(( ${LOGICAL_SEED_BASE:-1000} + i ))
  log "step $((i+1))/$NSTEPS: round$R over ${TASKS_ALL// /,}"
  JOB=""
  for _ in $(seq 1 20); do
    RAW=$(cd "$SAH" && timeout 60s env ROUND_ID="$R" OUT_TAG="$OUT_TAG" TASKS="$TASKS_ALL" \
      PROPOSER_SEED="$LOGICAL_PROPOSER_SEED" LOGICAL_ROUND_INDEX="$i" \
      K="${K:-8}" MAX_EVALS="${MAX_EVALS:-20}" EVAL_TIMEOUT="${EVAL_TIMEOUT:-420}" \
      FORCE_TOOL_FRAC="${FTF:-0.25}" SAH_MIN_ITERS="${SAH_MIN_ITERS:-0}" \
      SAH_ADV=v3 SAH_ANALYSIS="${USE_ANALYST:-1}" SAH_LEAK_NEUTRALIZE=1 \
      SAH_ANALYSIS_REQUIRED="${SAH_ANALYSIS_REQUIRED:-0}" \
      SAH_FIXED_INFERENCE_SLOTS="${SAH_FIXED_INFERENCE_SLOTS:-0}" \
      BASES_FILE="$bases" MPHI_PATH="${MPHI:-}" \
      SEED_PROGRAMS_FILE="$WS/best_programs.json" \
      FEEDBACK_FILE="$WS/task_feedback.json" \
      sbatch --parsable scripts/outer_round.sbatch 2>&1)
    JOB=$(echo "$RAW" | grep -oE '[0-9]{6,}' | tail -1)
    [ -n "$JOB" ] && break; sleep 60
  done
  [ -n "$JOB" ] || { log "submit failed: $(echo "$RAW"|tail -1)"; break; }
  log "  job $JOB"; wait_job "$JOB"

  [ -f "$RD/round.json" ] && [ ! -f "$RD/round_summary.json" ] && \
    (cd "$SAH/src" && python3 -m outer.rounds.outer_round collect --round-dir "$RD") >/dev/null 2>&1
  [ -f "$RD/round_summary.json" ] || { log "no summary — stop"; break; }

  # Re-check the transition explicitly so a stale/zero seed can never make the
  # context arm replace the fixed H2 with a worse candidate.
  python3 - "$prev_bases" "$RD/next_bases.json" <<'PYB'
import json, sys
prev_path, next_path = sys.argv[1:]
prev, nxt = json.load(open(prev_path)), json.load(open(next_path))
restored = []
for task, old in prev.items():
    new = nxt.get(task)
    if new is None or float(new.get("score", float("-inf"))) < float(old.get("score", float("-inf"))):
        nxt[task] = old
        restored.append(task)
json.dump(nxt, open(next_path, "w"), indent=1)
if restored:
    print("  validity ratchet restored fixed-H2 base for", ",".join(restored))
PYB

  # The collector owns the program transition.  Canonical proposer/context
  # arms use the exact same strict-single implementation; OUT_TAG makes this
  # file campaign-local, so copying it cannot import another route's state.
  [ -f "$OUT/best_programs.json" ] && \
    cp "$OUT/best_programs.json" "$WS/best_programs.json"
  # feedback must accumulate too — it is what turns the analyst on from round 2
  [ -f "$OUT/task_feedback.json" ] && cp "$OUT/task_feedback.json" "$WS/task_feedback.json" 2>/dev/null
  python3 - "$WS/task_feedback.json" <<'PY'
import json,sys,os
f=sys.argv[1]
if os.path.exists(f):
    d=json.load(open(f)); n=0
    for t,e in d.items():
        if isinstance(e,dict) and e.pop("analyst_note",None) is not None: n+=1
    if n: json.dump(d,open(f,"w"),indent=1); print(f"  stripped {n} curated note(s)")
PY
  bases="$RD/next_bases.json"
  python3 - "$RD/round_summary.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]))["groups"]
for t,v in sorted(g.items()):
    print(f"  {t:38s} best={v.get('best_score')} improved={v.get('improved')}")
PY
  log "  phi UNCHANGED (context-only ablation)"
  if [ -n "${POST_BATCH_HOOK:-}" ]; then
    ROUND_DIR="$RD" BATCH_INDEX="$i" ROUTE=context \
      bash "$POST_BATCH_HOOK" || log "  post-batch hook failed (non-fatal)"
  fi
done
log "context ablation done"
