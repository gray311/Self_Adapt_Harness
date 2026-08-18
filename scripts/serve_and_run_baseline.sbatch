#!/usr/bin/env bash
# Serve N replicas of Qwen3.5-9B (M0) on a GB200 node and run the M0+H2 baseline,
# sharded across replicas. Runs inside the aarch64 container (dgemma-core), which
# has numpy/scipy/sklearn/uv + the weave vLLM env; the worker adds nexau + jax/
# optax/orjson/cvxpy at runtime. Algotune tasks use $CODE_ROOT/AlgoTune (symlinked
# into raw/eft-aac2e79/AlgoTune).
#
# Submit:
#   source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
#   mkdir -p "$LOG_ROOT/slurm"; sbatch scripts/serve_and_run_baseline.sbatch
# Override via env: MAX_EVALS, N_REPLICAS, TASK_IDS.
#
#SBATCH --job-name=sah-h2-batch
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.err

set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

export IMAGE="${IMAGE:-/lustre/fsw/portfolios/av/users/yingzim/sqsh/dgemma-core-aarch64.sqsh}"
export REPO="$CODE_ROOT/self_adapt_harness"
export VLLM_ENV="${VLLM_ENV:-$ENV_ROOT/weave-qwen35-vllm/0.17.1}"
export MODEL_PATH="${MODEL_PATH:-$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
export SERVED_MODEL="${SERVED_MODEL:-qwen3.5-9b}"
export N_REPLICAS="${N_REPLICAS:-4}"
export MAX_EVALS="${MAX_EVALS:-20}"
# 10 CPU-runnable of the paper's 22: Math (5) + Algorithmic Heuristics (5).
export TASK_IDS="${TASK_IDS:-eft__math__circle_packing eft__math__hadamard_maximal_det eft__math__erdos_min_overlap eft__math__first_autocorr_ineq eft__math__second_autocorr_ineq eft__algotune__affine_transform_2d eft__algotune__convolve2d_full_fill eft__algotune__polynomial_real eft__algotune__psd_cone_projection eft__algotune__fft_convolution}"

export OUT_DIR="$RUN_ROOT/self_adapt_harness/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --export=ALL \
  --container-image="$IMAGE" \
  --container-mounts="/lustre:/lustre,/tmp:/tmp" \
  --container-workdir="$REPO" \
  bash "$REPO/scripts/_baseline_worker.sh"

echo "results: $OUT_DIR/summary_all.csv"
