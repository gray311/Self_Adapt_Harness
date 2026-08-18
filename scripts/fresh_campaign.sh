#!/usr/bin/env bash
# Single-task fresh campaign from BASE phi, generative genome (h2spec/1.0).
#   fresh_campaign.sh <task_id> <n_steps> <round_base> [force_tool_frac]
# Round r: propose (M_phi = latest merged phi, or BASE for step 1) with the
# generative genome -> rollout -> collect -> GRPO train next phi from prev ckpt
# -> merge -> repeat. Inheritance/feedback live in a task-local workspace so the
# main campaign is untouched. RL budget = n_steps (reset from scratch).
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
TASK="$1"; NSTEPS="$2"; RBASE="$3"; FTF="${4:-0.25}"; WS="${5:-$RUN_ROOT/self_adapt_harness/fresh_cp}"
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
OUT_TAG="${OUT_TAG:-}"
OUT="$RUN_ROOT/self_adapt_harness/outer${OUT_TAG:+-$OUT_TAG}"
BASE_PHI="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TAG=$(echo "$TASK" | sed 's/.*__//; s/_//g' | cut -c1-8)   # short per-task checkpoint tag
STAG_PREFIX="${STAG_PREFIX:-f_${TAG}}"
log(){ echo "[$(date -Is)] [fresh:${TASK##*__}] $*"; }

# Robust job wait: a transient squeue/slurmctld blip returns empty output and
# must NOT look like completion (observed: hadamard round742 — driver "finished"
# waiting while the job was mid-rollout, collected an empty round as best=None
# and moved on). When squeue says inactive, confirm via sacct; if sacct is also
# blank/blipped, keep waiting — the safe direction.
wait_job(){
  local J="$1" ST
  [ -n "$J" ] || return 0
  while :; do
    if squeue -j "$J" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; then
      sleep 150; continue
    fi
    ST=$(sacct -j "$J" -X -n -o State 2>/dev/null | head -1 | xargs)
    case "$ST" in PENDING|RUNNING|COMPLETING|"") sleep 60 ;; *) return 0 ;; esac
  done
}

bases="${RESUME_BASES:-$WS/round000_bases.json}"
[ -s "$bases" ] || { log "missing bases file: $bases"; exit 2; }
# RESUME_PHI/RESUME_CKPT: continue a phi lineage after a driver crash (e.g. the
# round742/763 squeue-blip incidents) instead of retraining from base.
prev_ckpt="${RESUME_CKPT:-}"       # empty => train from base on step 1
phi="${RESUME_PHI:-$BASE_PHI}"     # step 1 proposer = base unless resuming

for i in $(seq 0 $((NSTEPS - 1))); do
  [ -f "$WS/STOP" ] && { log "STOP flag — exiting"; break; }
  if [ -n "${RUNTIME_SOURCE_MANIFEST:-}" ]; then
    python3 "$SAH/scripts/runtime/provenance.py" verify \
      --manifest "$RUNTIME_SOURCE_MANIFEST" || exit 43
  fi
  R=$((RBASE + i)); RD="$OUT/round$(printf '%03d' "$R")"
  STAG=$(printf "%s_%02d" "$STAG_PREFIX" "$i")
  LOGICAL_PROPOSER_SEED=$(( ${LOGICAL_SEED_BASE:-1000} + i ))
  log "step $((i+1))/$NSTEPS: round$R propose (phi=$(basename "$phi")) ftf=$FTF"

  JOB=""
  for _ in $(seq 1 30); do
    RAW=$(cd "$SAH" && env ROUND_ID="$R" OUT_TAG="$OUT_TAG" TASKS="$TASK" K="${K:-8}" MAX_EVALS="${MAX_EVALS:-20}" \
      PROPOSER_SEED="$LOGICAL_PROPOSER_SEED" LOGICAL_ROUND_INDEX="$i" \
      FORCE_TOOL_FRAC="$FTF" EVAL_TIMEOUT="${EVAL_TIMEOUT:-180}" EVAL_REPEATS="${EVAL_REPEATS:-1}" SAH_MIN_ITERS="${SAH_MIN_ITERS:-0}" \
      SAH_ADV="${SAH_ADV:-v3}" SAH_HIST_LAMBDA="${SAH_HIST_LAMBDA:-0.3}" SAH_ALPHA="${SAH_ALPHA:-0.3}" \
      SAH_SEQUENTIAL="${SAH_SEQUENTIAL:-0}" SAH_SEQ_MAX_SHARED="${SAH_SEQ_MAX_SHARED:-6}" \
      SAH_ANALYSIS="${SAH_ANALYSIS:-0}" SAH_LEAK_NEUTRALIZE="${SAH_LEAK_NEUTRALIZE:-1}" \
      SAH_ANALYSIS_REQUIRED="${SAH_ANALYSIS_REQUIRED:-0}" \
      SAH_FIXED_INFERENCE_SLOTS="${SAH_FIXED_INFERENCE_SLOTS:-0}" \
      SAH_CHAMPION_ISOLATION="${SAH_CHAMPION_ISOLATION:-1}" \
      BASES_FILE="$bases" MPHI_PATH="$phi" \
      SEED_PROGRAMS_FILE="$WS/best_programs.json" \
      FEEDBACK_FILE="$WS/task_feedback.json" \
      sbatch --parsable scripts/outer_round.sbatch 2>&1)
    JOB=$(echo "$RAW" | grep -oE '[0-9]{6,}' | tail -1)
    [ -n "$JOB" ] && break; sleep 60
  done
  [ -n "$JOB" ] || { log "submit failed: $(echo "$RAW" | tail -2 | tr '\n' ' ')"; break; }
  log "  job $JOB"
  wait_job "$JOB"

  if [ ! -f "$RD/grpo_batch.jsonl" ] && [ -f "$RD/round.json" ]; then
    (cd "$SAH/src" && python3 -m outer.rounds.outer_round collect --round-dir "$RD") >/dev/null 2>&1
  fi
  [ -f "$RD/round_summary.json" ] || { log "no summary — stop"; break; }
  python3 "$SAH/scripts/runtime/sanitize_grpo_batch.py" "$RD" >/dev/null 2>&1
  # sync inheritance + feedback into the task-local workspace.
  # NO_INHERIT=1 breaks the program ratchet on purpose: every round starts from
  # the ORIGINAL seed instead of the incumbent. Use it to escape a basin that the
  # ratchet has locked in — observed on circle packing, where all K candidates
  # inherited the same 2.502 grid, scored identically to base, and produced a
  # zero-variance group ("no_signal(true-plateau)"), so phi never trained.
  if [ "${NO_INHERIT:-0}" = "1" ]; then
    rm -f "$WS/best_programs.json"
  else
    [ -f "$OUT/best_programs.json" ] && cp "$OUT/best_programs.json" "$WS/best_programs.json" 2>/dev/null || true
  fi
  [ -f "$OUT/task_feedback.json" ] && cp "$OUT/task_feedback.json" "$WS/task_feedback.json" 2>/dev/null || true
  if [ "${NO_CURATED_NOTES:-0}" = "1" ] && [ -f "$WS/task_feedback.json" ]; then
    python3 - "$WS/task_feedback.json" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.load(open(path))
removed = 0
for row in payload.values():
    if isinstance(row, dict) and row.pop("analyst_note", None) is not None:
        removed += 1
with open(path, "w") as handle:
    json.dump(payload, handle, indent=1)
if removed:
    print(f"  stripped {removed} curated analyst note(s)")
PY
  fi
  SUM=$(python3 -c "
import json
g=json.load(open('$RD/round_summary.json'))['groups']['$TASK']
print('base=%.5g best=%s improved=%s'%(g['base_score'], g['best_score'], g['improved']))")
  DIMS=$(python3 -c "
import json
d=json.load(open('$RD/round.json'))
from collections import Counter
cnt=Counter()
for c in d['per_task']['$TASK']['candidates']:
    for f in c.get('changed_fields',[]):
        if f.startswith('new_'): cnt[f.split('.')[0]]+=1
print(dict(cnt))")
  log "  $SUM | gen_dims=$DIMS"
  bases="$RD/next_bases.json"

  # ---- GRPO train next phi ----
  V=$(python3 -c "import json;d=json.load(open('$RD/round.json'));print(sum(1 for c in d['per_task']['$TASK']['candidates'] if c['valid']))")
  # plateau-gated commit cadence (campaign_config: training.plateau_rounds):
  # PR=1 (default) trains every round. PR>1 skips training on non-improving
  # rounds until PR consecutive stalls accumulate, then commits one update —
  # Adaptive's confirmed-plateau update schedule. Improving rounds always train
  # and reset the stall counter.
  PR="${PLATEAU_ROUNDS:-1}"
  IMPROVED=$(python3 -c "import json;print(1 if json.load(open('$RD/round_summary.json'))['groups']['$TASK'].get('improved') else 0)" 2>/dev/null || echo 0)
  if [ "$IMPROVED" = "1" ]; then stall=0; else stall=$((${stall:-0} + 1)); fi
  do_train=1
  if [ "${SKIP_FINAL_TRAIN:-0}" = 1 ] && [ "$i" -eq $((NSTEPS - 1)) ]; then
    do_train=0
    log "  final measurement batch: skip an unevaluated trailing proposer update"
  fi
  if [ "$PR" -gt 1 ] && [ "$IMPROVED" != "1" ] && [ "$stall" -lt "$PR" ]; then
    do_train=0; log "  plateau-gate: stall $stall/$PR, deferring training"
  fi
  # NO_TRAIN=1 freezes the proposer's weights for the whole campaign. This is the
  # \method (context) ablation: the harness still evolves round to round, but only
  # through what the proposer READS (incumbent + experience digest + analyst
  # brief), never through a gradient. Isolates context adaptation from weight
  # adaptation.
  if [ "${NO_TRAIN:-0}" = "1" ]; then
    do_train=0; log "  NO_TRAIN=1: phi frozen (context-only ablation)"
  fi
  min_trainable="${MIN_TRAINABLE_ROWS:-4}"
  trainable=0
  if [ "$do_train" = "1" ]; then
    if python3 "$SAH/src/training/grpo_to_replay.py" \
        --rounds "$RD" --out "$RD/replay.jsonl" > "$RD/replay_preflight.log" 2>&1; then
      trainable=$(wc -l < "$RD/replay.jsonl")
    else
      log "  replay preflight failed — skip training"
    fi
  fi
  if [ "$trainable" -ge "$min_trainable" ] && [ "$do_train" = "1" ]; then
    stall=0
    cd "$SAH"
    # Every task has its own controller and the controllers may share one CPU
    # node.  A fixed /tmp filename lets concurrent proposer arms overwrite one
    # another's parsed train/merge job IDs, so keep the submission receipt in
    # the immutable round directory instead.
    TRAIN_SUBMIT_LOG="$RD/train_submit.log"
    if [ -z "$prev_ckpt" ]; then
      KL_COEF="${KL_COEF:-0.05}" NUM_EPOCH="${NUM_EPOCH:-3}" bash scripts/train_mphi_step.sh "$RD" "$STAG" > "$TRAIN_SUBMIT_LOG" 2>&1
    else
      KL_COEF="${KL_COEF:-0.05}" NUM_EPOCH="${NUM_EPOCH:-3}" bash scripts/train_mphi_step.sh "$RD" "$STAG" "$prev_ckpt" > "$TRAIN_SUBMIT_LOG" 2>&1
    fi
    T=$(grep -oP 'train job: \K[0-9]+' "$TRAIN_SUBMIT_LOG"); M=$(grep -oP 'merge job: \K[0-9]+' "$TRAIN_SUBMIT_LOG")
    if [ -n "$T" ]; then
      wait_job "$T"
      if [ "$(sacct -j "$T" -X -n -o State|head -1|xargs)" = "COMPLETED" ]; then
        wait_job "$M"
        MERGED="$MODEL_ROOT/exports/self_adapt_harness/mphi_$STAG"
        if ls "$MERGED"/*.safetensors >/dev/null 2>&1; then
          phi="$MERGED"; prev_ckpt="$MODEL_ROOT/checkpoints/self_adapt_harness/mphi_$STAG"
          log "  trained -> mphi_$STAG"
        fi
      fi
    fi
  elif [ "$do_train" = "1" ]; then
    log "  insufficient trainable H1 rows ($trainable < $min_trainable; valid=$V) — skip training, keep phi"
  fi
  if [ -n "${POST_BATCH_HOOK:-}" ]; then
    ROUND_DIR="$RD" BATCH_INDEX="$i" ROUTE=proposer \
      bash "$POST_BATCH_HOOK" || log "  post-batch hook failed (non-fatal)"
  fi
done
log "fresh campaign done"
