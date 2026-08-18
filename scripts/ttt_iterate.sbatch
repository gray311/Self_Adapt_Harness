#!/usr/bin/env bash
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --job-name=ttt-iter
#SBATCH --output=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.err
#
# Iterative test-time training on the EXECUTOR -- the real TTT-Discover loop:
#
#   for each round:  generate K solutions with the current executor
#                    -> score them -> train the executor on its own best
#                    -> serve the updated executor -> repeat
#
# The harness is the FIXED initial harness and no proposer ever runs, so the only
# thing that changes across rounds is the executor's weights. Every training row
# is a program this same model produced: no external solution, no stronger model.
#
# A checkpoint of (cumulative rollouts, best score) is appended each round, which
# is exactly the score-vs-compute curve for this arm.
#
# Env: TASK  OUT_DIR  [ROUNDS=4] [K=32] [MAX_EVALS=20] [EVAL_TIMEOUT=240]
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"; W="$CODE_ROOT/Weave_v2"
export VLLM_ENV="${VLLM_ENV:-$ENV_ROOT/weave-qwen35-vllm/0.17.1}"
BASE_HF="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
: "${TASK:?}"; : "${OUT_DIR:?}"
ROUNDS="${ROUNDS:-4}"; K="${K:-64}"; MAX_EVALS="${MAX_EVALS:-20}"; EVAL_TIMEOUT="${EVAL_TIMEOUT:-240}"
# TTT-Discover Table 9: 512 rollouts/step as 8 groups x 64, 50 steps, LoRA r32,
# Adam lr 4e-5, KL 0.1, sampling temperature 1.0, entropic-utility objective.
# We keep their per-group size (64), rank, lr, KL and temperature; we cannot
# afford 8 groups x 50 steps on our GPU share, so the budget is smaller and the
# figure states both budgets explicitly.
TTT_TEMP="${TTT_TEMP:-1.0}"
SERVED_MODEL="qwen3.5-9b"; mkdir -p "$OUT_DIR"
log(){ echo "[$(date -Is)] [ttt-iter] $*"; }
case "$TASK" in *ahc*) export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS=12 \
  AHC_CACHE_DIR="$SAH/ahc_work/cache";; esac
export VLLM_USE_FLASHINFER_SAMPLER=0

export IMAGE="${IMAGE:-/lustre/fsw/portfolios/av/users/yingzim/sqsh/dgemma-core-aarch64.sqsh}"
export TASK OUT_DIR ROUNDS K MAX_EVALS EVAL_TIMEOUT TTT_TEMP SERVED_MODEL
srun --nodes=1 --ntasks=1 --cpus-per-task=64 --export=ALL \
  --container-image="$IMAGE" --container-mounts="/lustre:/lustre,/tmp:/tmp" \
  --container-workdir="$CODE_ROOT/self_adapt_harness" \
  bash "$CODE_ROOT/self_adapt_harness/scripts/_ttt_iter_worker.sh"
