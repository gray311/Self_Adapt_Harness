#!/usr/bin/env bash
# Serialize live ledger/HTML refreshes from the twelve independent pipelines.
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"
RESULT="$SAH/results/reward_route_inference16_effects.json"
LOG="$SAH/results/reward_route_inference16_refresh.log"

exec 9>"$SAH/results/reward_route_inference16_refresh.lock"
flock 9
{
  echo "[$(date -Is)] refresh route=${ROUTE:-manual} batch=${BATCH_INDEX:-manual} round=${ROUND_DIR:-manual}"
  cd "$SAH"
  PYTHONPATH=.:src python3 scripts/analysis/reward_route/effects.py
  python3 scripts/analysis/reward_route/render_live.py
  status=$(python3 - "$RESULT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["status"])
PY
)
  if [ "$status" = complete ]; then
    MPLCONFIGDIR=/tmp/mpl-rri16 PYTHONPATH=.:src \
      python3 scripts/analysis/figures/reward_route_inference16.py
    PYTHONPATH=.:src python3 scripts/analysis/reward_route/collect_endpoint_programs.py \
      --data papers/figures/reward_route_inference16_1x4_data.json \
      --run-root "$RUN_ROOT" \
      --out results/reward_route_inference16_endpoint_cases.json
    PYTHONPATH=.:src python3 scripts/analysis/audits/reward_route_inference16.py \
      --view papers/figures/reward_route_inference16_1x4_data.json \
      --out results/reward_route_inference16_prevalidation_audit.json
    echo "search curves complete; awaiting N>=5 endpoint revalidation before final binding"
  fi
} >> "$LOG" 2>&1
