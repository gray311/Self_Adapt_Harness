#!/usr/bin/env bash
# Cross-task proposer transfer: does a proposer trained on task i produce good
# harnesses for task j?
#
#   cross_task_transfer.sh <round_base> [n_parallel]
#
# One round per SOURCE adapter, each round covering ALL 11 target tasks, so row i
# of the transfer matrix comes from a single job.  Zero-shot: the adapter is used
# as-is, never updated on the target.  K rollouts per cell (not 1) because a
# single solution trajectory is too noisy to read a transfer effect from.
#
# Row phi_0 (the untrained base proposer) is the baseline every other row is
# scored against.
#
# Leakage discipline: every proposer here is a LoRA over the same frozen base as
# the executor; no stronger model, no curated notes (stripped below), analysis
# off so the only cross-task signal is what the adapter itself carries.
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
OUT="$RUN_ROOT/self_adapt_harness/outer"
E="$MODEL_ROOT/exports/self_adapt_harness"
RBASE="${1:?usage: cross_task_transfer.sh <round_base> [n_parallel]}"
NPAR="${2:-2}"
WS="$RUN_ROOT/self_adapt_harness/cross_task"
mkdir -p "$WS"
log(){ echo "[$(date -Is)] [xtask] $*"; }

TASKS_ALL="eft__math__erdos_min_overlap eft__math__first_autocorr_ineq eft__math__second_autocorr_ineq eft__math__circle_packing eft__math__hadamard_maximal_det eft__ahc_simpletes__ahc039 eft__ahc_simpletes__ahc058 adrs__eplb adrs__prism adrs__llm_sql adrs__txn_scheduling"

# source rows: phi_0 (base) first, then the 11 per-task adapters
SOURCES="BASE mphi_f_erdosmin_09 mphi_f_firstaut_11 mphi_f_secondau_10 mphi_f_circlepa_03 mphi_f_hadamard_10 mphi_f_ahc039_07 mphi_f_ahc058_07 mphi_f_eplb_04 mphi_f_prism_05 mphi_f_llmsql_05 mphi_f_txnsched_03"

# all rows start from the SAME fixed initial harness and the seed program, so a
# cell measures the adapter's proposal quality, not an inherited incumbent.
bases="$WS/round000_bases.json"
python3 - "$bases" "$SAH" "$TASKS_ALL" <<'PY'
import json,sys
out,sah,tasks=sys.argv[1],sys.argv[2],sys.argv[3].split()
json.dump({t:{"package":f"{sah}/src/inner/harness","score":0.0} for t in tasks}, open(out,"w"), indent=1)
PY
rm -f "$WS/best_programs.json" "$WS/task_feedback.json"   # zero-shot: no inheritance, no history

i=0
for SRC in $SOURCES; do
  R=$((RBASE+i)); i=$((i+1))
  if [ "$SRC" = "BASE" ]; then MP=""; else MP="$E/$SRC"; fi
  [ -n "$MP" ] && [ ! -d "$MP" ] && { log "skip $SRC (missing)"; continue; }
  log "row $i: source=$SRC -> round$R over all 11 targets"
  RAW=$(cd "$SAH" && env ROUND_ID="$R" TASKS="$TASKS_ALL" \
    K="${K:-6}" MAX_EVALS="${MAX_EVALS:-15}" EVAL_TIMEOUT="${EVAL_TIMEOUT:-240}" \
    FORCE_TOOL_FRAC=0.25 SAH_ADV=v3 SAH_ANALYSIS=0 SAH_LEAK_NEUTRALIZE=1 \
    BASES_FILE="$bases" MPHI_PATH="$MP" \
    SEED_PROGRAMS_FILE="$WS/none.json" \
    FEEDBACK_FILE="$WS/none_fb.json" \
    sbatch --parsable scripts/outer_round.sbatch 2>&1)
  J=$(echo "$RAW" | grep -oE '[0-9]{6,}' | tail -1)
  [ -n "$J" ] || { log "  submit failed: $(echo "$RAW"|tail -1)"; continue; }
  echo "$SRC $R $J" >> "$WS/rows.txt"
  MYJOBS="${MYJOBS:+$MYJOBS }$J"
  log "  job $J"
  # keep at most NPAR of THIS experiment's rows in flight (other campaigns of
  # ours are running concurrently and must not be counted here)
  while :; do
    live=0
    for j in $MYJOBS; do
      s=$(squeue -j "$j" -h -o %T 2>/dev/null | head -1)
      case "$s" in PENDING|RUNNING|COMPLETING) live=$((live+1)) ;; esac
    done
    [ "$live" -lt "$NPAR" ] && break
    sleep 120
  done
done
log "all rows submitted; collect with scripts/analysis/collect/cross_task.py"
