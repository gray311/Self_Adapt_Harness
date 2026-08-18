#!/usr/bin/env bash
# Launch fresh from-base generative campaigns for ALL tasks, bounded by a GPU
# budget. Each task gets its own workspace (fresh_all/<task>), a disjoint round
# range, and a task-tagged phi lineage. A task-campaign holds one 4-GPU serving
# job at a time, so MAX_TASKS concurrent campaigns ~= GPU_BUDGET/4.
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"
FRESH="$RUN_ROOT/self_adapt_harness/fresh_all"
NSTEPS="${NSTEPS:-8}"
FTF="${FTF:-0.25}"
GPU_BUDGET="${GPU_BUDGET:-28}"     # keep <32; each campaign uses one 4-GPU job
MAX_TASKS=$(( GPU_BUDGET / 4 ))

# task -> round base (disjoint 20-wide ranges) and native-eval env
TASKS=(
  eft__math__circle_packing:310
  eft__math__hadamard_maximal_det:330
  eft__math__erdos_min_overlap:350
  eft__math__first_autocorr_ineq:370
  eft__math__second_autocorr_ineq:390
  adrs__prism:410
  adrs__eplb:430
  adrs__txn_scheduling:450
  adrs__llm_sql:470
  eft__ahc_simpletes__ahc039:490
  eft__ahc_simpletes__ahc058:510
)

echo "[launch] $(date -Is) MAX_TASKS=$MAX_TASKS (GPU budget $GPU_BUDGET) NSTEPS=$NSTEPS FTF=$FTF"
running=0
for entry in "${TASKS[@]}"; do
  task="${entry%%:*}"; rbase="${entry##*:}"
  ws="$FRESH/$task"
  # AHC needs native aarch64 eval env exported into the campaign
  extra_env=""
  if [[ "$task" == *ahc* ]]; then
    export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS=12 \
           AHC_CACHE_DIR="$SAH/ahc_work/cache"
  fi
  # throttle to the GPU budget
  while [ "$(ls "$FRESH"/*/RUNNING 2>/dev/null | wc -l)" -ge "$MAX_TASKS" ]; do sleep 120; done
  touch "$ws/RUNNING"
  ( bash "$SAH/scripts/fresh_campaign.sh" "$task" "$NSTEPS" "$rbase" "$FTF" "$ws" \
      > "$ws/driver.log" 2>&1; rm -f "$ws/RUNNING" ) &
  echo "[launch] started $task (round$rbase+, ws=$ws)"
  sleep 20
done
wait
echo "[launch] all task-campaigns finished $(date -Is)"
