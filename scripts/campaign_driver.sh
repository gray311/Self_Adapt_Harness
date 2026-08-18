#!/usr/bin/env bash
# Continuous campaign driver: iterate outer steps + phi versions until every
# finch target is ACHIEVED (or $OUT/DRIVER_STOP exists).
#
# Loop: adopt-or-submit step job -> wait -> collect(v2) -> sanitize ->
#       adaptive train next phi version (KL 0.1, 2 epochs, ARCHIVE_MIX) ->
#       merge -> repeat with the updated phi.
# Task policy (SOTA-targeted): cycle tasks still below sota_combined, ordered
# by relative gap (closest first); cascade K=16 default; stops when all beat SOTA.
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"
OUT="$RUN_ROOT/self_adapt_harness/outer"
LOG="$OUT/driver.log"
log(){ echo "[$(date -Is)] $*" | tee -a "$LOG"; }

sbatch_retry(){
  local out
  for _ in $(seq 1 30); do
    if out=$("$@" 2>&1); then echo "$out" | tail -1; return 0; fi
    sleep 60
  done
  return 1
}

latest_phi(){ # highest merged export
  ls -d "$MODEL_ROOT/exports/self_adapt_harness"/mphi_s* 2>/dev/null \
    | sort -V | while read -r d; do [ -f "$d/model.safetensors" ] && echo "$d"; done | tail -1
}
latest_ckpt(){ ls -d "$MODEL_ROOT/checkpoints/self_adapt_harness"/mphi_s* 2>/dev/null | sort -V | tail -1; }
next_round(){ ls -d "$OUT"/round0* 2>/dev/null | sed 's/.*round0*//' | sort -n | tail -1 | awk '{print $1+1}'; }
next_stag(){ latest_ckpt | grep -oE 's[0-9]+$' | sed 's/s//' | awk '{printf "s%03d", $1+1}'; }

pick_task(){ # prints "task_id|extra sbatch env" or DONE — SOTA-targeted
  python3 - "$OUT" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
root = os.environ["CODE_ROOT"] + "/self_adapt_harness"
ft = json.load(open(f"{root}/results/finch_targets.json"))["targets"]
try:
    bp = json.load(open(f"{out}/best_programs.json"))
except Exception:
    bp = {}
pending = []
for t, v in ft.items():
    sota = v.get("sota_combined")
    if sota is None:
        continue
    cur = max(filter(None, [(bp.get(t) or {}).get("score"), v.get("current_best"), 0.0]))
    if cur <= sota:
        pending.append(((sota - cur) / abs(sota), t))
pending.sort()
if not pending:
    print("DONE"); raise SystemExit
n_rounds = len(glob.glob(f"{out}/round0*"))
task = pending[n_rounds % len(pending)][1]
P = {  # per-task params: cascade default K=16/5ev-screen/top4-full
    "adrs__eplb":                      "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=30 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "eft__math__erdos_min_overlap":    "K=8 MAX_EVALS=60",
    "eft__math__first_autocorr_ineq":  "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=40 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "eft__math__second_autocorr_ineq": "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=40 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "eft__math__hadamard_maximal_det": "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=40 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "eft__math__circle_packing":       "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=40 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "adrs__prism":                     "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=30 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "adrs__txn_scheduling":            "K=16 SCREEN_EVALS=5 PROMOTE=4 MAX_EVALS=30 SEED_PROGRAMS_FILE={out}/best_programs.json",
    "adrs__llm_sql":                   "K=8 MAX_EVALS=40 EVAL_TIMEOUT=420 SEED_PROGRAMS_FILE={out}/best_programs.json",
}
print(f"{task}|{P[task].format(out=out)}")
PY
}

log "driver start; phi=$(basename "$(latest_phi)")"
while true; do
  [ -f "$OUT/DRIVER_STOP" ] && { log "DRIVER_STOP found — exiting"; break; }

  JOB=$(squeue -u "$USER" -h -n sah-outer -o %i | head -1)
  if [ -z "$JOB" ]; then
    PICK=$(pick_task)
    [ "$PICK" = "DONE" ] && { log "ALL TARGETS ACHIEVED — driver done"; break; }
    TASK="${PICK%%|*}"; EXTRA="${PICK##*|}"
    R=$(next_round); PHI=$(latest_phi)
    PREV_RD=$(ls -d "$OUT"/round0* | sort -V | tail -1)
    log "submit round$R task=$TASK phi=$(basename "$PHI") extra=[$EXTRA]"
    JOB=""
    for _ in $(seq 1 30); do
      # env runs the real sbatch binary (a shell function can't be exec'd by env)
      RAW=$(cd "$SAH" && env ROUND_ID="$R" TASKS="$TASK" $EXTRA \
        BASES_FILE="$PREV_RD/next_bases.json" MPHI_PATH="$PHI" \
        FEEDBACK_FILE="$OUT/task_feedback.json" \
        sbatch --parsable scripts/outer_round.sbatch 2>&1)
      JOB=$(echo "$RAW" | grep -oE '[0-9]{6,}' | tail -1)
      [ -n "$JOB" ] && break
      log "  sbatch retry ($(echo "$RAW" | tail -1 | cut -c1-80))"
      sleep 60
    done
    [ -n "$JOB" ] || { log "submit failed after 30 tries; pausing 10m"; sleep 600; continue; }
    log "  job $JOB"
  else
    log "adopting running step job $JOB"
  fi

  while squeue -j "$JOB" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do sleep 180; done
  RD=$(ls -d "$OUT"/round0* | sort -V | tail -1)
  log "job $JOB ended ($(sacct -j "$JOB" -X -n -o State | head -1 | xargs)); round dir $RD"

  if [ ! -f "$RD/grpo_batch.jsonl" ] && [ -f "$RD/round.json" ]; then
    (cd "$SAH/src" && python3 -m outer.rounds.outer_round collect --round-dir "$RD") >> "$LOG" 2>&1
  fi
  [ -f "$RD/round_summary.json" ] || { log "no summary for $RD — skipping"; sleep 60; continue; }
  python3 "$SAH/scripts/runtime/sanitize_grpo_batch.py" "$RD" >> "$LOG" 2>&1
  python3 - "$RD" <<'PY' >> "$LOG" 2>&1
import json, sys
d = json.load(open(f"{sys.argv[1]}/round_summary.json"))
for tid, g in d["groups"].items():
    print(f"  {tid}: base={g['base_score']:.6g} best={g['best_score']} "
          f"improved={g['improved']} mode={g.get('adv_mode')}")
PY

  V=$(python3 -c "
import json
d=json.load(open('$RD/round.json'))
print(sum(1 for pt in d['per_task'].values() for c in pt['candidates'] if c['valid']))")
  NK=$(python3 -c "
import json
d=json.load(open('$RD/round.json'))
print(sum(len(pt['candidates']) for pt in d['per_task'].values()))")
  if [ "$V" -ge $((NK / 2)) ]; then
    STAG=$(next_stag); CKPT=$(latest_ckpt)
    log "train $STAG from $(basename "$CKPT") (V=$V/$NK, archive-mix on)"
    KL_COEF=0.1 NUM_EPOCH=2 ARCHIVE_MIX=4 \
      bash "$SAH/scripts/train_mphi_step.sh" "$RD" "$STAG" "$CKPT" > /tmp/driver_train.txt 2>&1
    cat /tmp/driver_train.txt >> "$LOG"
    T=$(grep -oP 'train job: \K[0-9]+' /tmp/driver_train.txt)
    M=$(grep -oP 'merge job: \K[0-9]+' /tmp/driver_train.txt)
    if [ -n "$T" ]; then
      while squeue -j "$T" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do sleep 120; done
      TS=$(sacct -j "$T" -X -n -o State | head -1 | xargs); log "train $STAG: $TS"
      if [ "$TS" = "COMPLETED" ] && [ -n "$M" ]; then
        while squeue -j "$M" -h -o '%T' 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do sleep 60; done
        log "merged: $(basename "$(latest_phi)")"
      fi
    else
      log "train not submitted (no signal rows?) — continuing with current phi"
    fi
  else
    log "degenerate group (V=$V/$NK) — skip training"
  fi
done
log "driver exit"
