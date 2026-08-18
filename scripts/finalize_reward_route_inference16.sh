#!/usr/bin/env bash
# Bind complete endpoint validation to the authoritative main-result registry.
# This does not submit jobs; run it only after
# scripts/analysis/reward_route/revalidate_endpoints.py.
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"
VIEW="${VIEW:-$SAH/papers/figures/reward_route_inference16_1x4_data.json}"
VALIDATION="${VALIDATION:-$SAH/results/reward_route_inference16_endpoint_validation.json}"
BINDING="${BINDING:-$SAH/results/reward_route_inference16_main_results.json}"

python3 "$SAH/scripts/analysis/reward_route/bind_main_results.py" \
  --validation "$VALIDATION" \
  --human "$SAH/results/human_best_references.json" \
  --out "$BINDING"
python3 "$SAH/scripts/analysis/audits/reward_route_inference16.py" \
  --view "$VIEW" --validation "$VALIDATION" --main-binding "$BINDING" \
  --out "$SAH/results/reward_route_inference16_final_audit.json"
echo "final inference16 results are bound to endpoint revalidation: $BINDING"
