#!/usr/bin/env bash
# 24h Finch sprint: per-task ratchet loop (no training — fixed phi, pure hunting).
#   sprint_task.sh <task_id> <finch_target> <round_base> <max_iters> [extra env...]
# Each iteration: submit outer round -> wait -> salvage-collect -> check target.
# Round ids = round_base, round_base+1, ... (disjoint ranges across parallel sprints).
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
TASK="$1"; TARGET="$2"; RBASE="$3"; MAXIT="$4"; shift 4
EXTRA=("$@")
SAH="$CODE_ROOT/self_adapt_harness"
OUT="$RUN_ROOT/self_adapt_harness/outer"
PHI="$MODEL_ROOT/exports/self_adapt_harness/mphi_s035"
BASES="$OUT/round041/next_bases.json"
log(){ echo "[$(date -Is)] [sprint:${TASK##*__}] $*"; }

for i in $(seq 0 $((MAXIT - 1))); do
  [ -f "$OUT/SPRINT_STOP" ] && { log "SPRINT_STOP — exiting"; break; }
  R=$((RBASE + i))
  RD="$OUT/round$(printf '%03d' "$R")"
  log "iter $((i+1))/$MAXIT: submit round$R (bases=$(basename "$(dirname "$BASES")"))"
  JOB=""
  for _ in $(seq 1 30); do
    RAW=$(cd "$SAH" && env ROUND_ID="$R" TASKS="$TASK" "${EXTRA[@]}" \
      BASES_FILE="$BASES" MPHI_PATH="$PHI" \
      SEED_PROGRAMS_FILE="$OUT/best_programs.json" \
      FEEDBACK_FILE="$OUT/task_feedback.json" \
      sbatch --parsable scripts/outer_round.sbatch 2>&1)
    JOB=$(echo "$RAW" | grep -oE '[0-9]{6,}' | tail -1)
    [ -n "$JOB" ] && break
    sleep 60
  done
  [ -n "$JOB" ] || { log "submit failed; abort sprint"; break; }
  log "  job $JOB"
  while squeue -j "$JOB" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do sleep 180; done
  log "  job $JOB ended: $(sacct -j "$JOB" -X -n -o State | head -1 | xargs)"
  if [ ! -f "$RD/grpo_batch.jsonl" ] && [ -f "$RD/round.json" ]; then
    (cd "$SAH/src" && python3 -m outer.rounds.outer_round collect --round-dir "$RD") 2>&1 | tail -3
  fi
  [ -f "$RD/round_summary.json" ] || { log "  no summary — continuing"; BASES="$BASES"; continue; }
  BEST=$(python3 -c "
import json
d = json.load(open('$RD/round_summary.json'))
g = d['groups'].get('$TASK') or {}
print(g.get('best_score') or 0)")
  log "  round$R best=$BEST (target $TARGET)"
  BASES="$RD/next_bases.json"
  HIT=$(python3 -c "print(1 if float('$BEST') > float('$TARGET') else 0)")
  if [ "$HIT" = "1" ]; then
    log "  *** FINCH BROKEN: $BEST > $TARGET ***"
    break
  fi
done
log "sprint done"
