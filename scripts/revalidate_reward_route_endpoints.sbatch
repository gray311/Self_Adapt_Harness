#!/usr/bin/env bash
# Evaluator-only endpoint validation on a CPU node.  No model server or GPU is
# used.  CASES_FILE may contain all 21 final cases or a provisional subset.
# Required: CASES_FILE OUT_FILE. Optional: N_RUNS, TOUCH_COMPLETE, EXPECTED_CASES.
#SBATCH --job-name=sah-endpoint-reval
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=cpu
#SBATCH --qos=cpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=230G
#SBATCH --time=7-00:00:00
#SBATCH --output=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.err

set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

: "${CASES_FILE:?set CASES_FILE to an endpoint-program manifest}"
: "${OUT_FILE:?set OUT_FILE to the validation JSON path}"
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/av/users/yingzim/sqsh/dgemma-core-aarch64.sqsh}"
N_RUNS="${N_RUNS:-5}"
TOUCH_COMPLETE="${TOUCH_COMPLETE:-0}"
EXPECTED_CASES="${EXPECTED_CASES:-21}"
VALIDATION_WORKERS="${VALIDATION_WORKERS:-4}"
mkdir -p "$(dirname "$OUT_FILE")"
export CASES_FILE OUT_FILE SAH N_RUNS TOUCH_COMPLETE EXPECTED_CASES VALIDATION_WORKERS

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --export=ALL \
  --container-image="$IMAGE" --container-mounts="/lustre:/lustre,/tmp:/tmp" \
  --container-workdir="$SAH" bash -s <<'WORKER'
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
export UV_BREAK_SYSTEM_PACKAGES=1
export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS=12
export AHC_CACHE_DIR="$SAH/ahc_work/cache"
uv pip install --system jax optax orjson cvxpy \
  > "$(dirname "$OUT_FILE")/validation_pip.log" 2>&1
python3 "$SAH/scripts/analysis/reward_route/revalidate_endpoints.py" \
  --cases "$CASES_FILE" --out "$OUT_FILE" --n-runs "$N_RUNS" --timeout 420 \
  --workers "$VALIDATION_WORKERS"
if [ "$TOUCH_COMPLETE" = 1 ]; then
  python3 - "$OUT_FILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get("status") == "complete"
assert p.get("all_runs_valid") is True
assert len(p.get("case_results") or {}) == int(__import__("os").environ["EXPECTED_CASES"])
PY
  touch "$(dirname "$OUT_FILE")/CANONICAL_COMPLETE"
fi
WORKER
