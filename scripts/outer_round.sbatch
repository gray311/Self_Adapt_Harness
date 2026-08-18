#!/usr/bin/env bash
# One outer-loop STEP (one task) on a GB200 node — the online GRPO cadence:
#   task tau: M_phi + H1(tau) -> K=8 candidate H2 -> M0 + each H2 -> 8 rollouts
#   on tau (<=20 evals each) -> 8 rewards (one GRPO group) -> grpo_batch.jsonl.
#   Then: train phi on this group (src/training), merge, and launch the next
#   step (next task / same task) against the updated M_phi.
#
# Submit:
#   source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
#   mkdir -p "$LOG_ROOT/slurm"
#   ROUND_ID=1 TASKS=eft__math__circle_packing sbatch scripts/outer_round.sbatch
# Env knobs: ROUND_ID TASKS K MAX_EVALS EVAL_TIMEOUT ROLLOUT_WALL_TIMEOUT
#            BASES_FILE MODEL_PATH PROPOSER_SEED OUT_TAG ROLLOUT_PAR PROPOSE_PAR
#            SAH_FIXED_INFERENCE_SLOTS SAH_ANALYSIS_REQUIRED
#   TASKS:        the task for this step (space-separated list allowed, default CP).
#   BASES_FILE:   per-task current-best-H2 registry (default: initial H2 + static
#                 baseline). For round r>1 use the previous round's next_bases.json.
#   MODEL_PATH:   serving checkpoint. Round 1 = frozen base (M_phi==M0 weights).
#                 After GRPO, point at the merged M_phi checkpoint (executor calls
#                 still go to the same served model name — see src/training/README).
#
#SBATCH --job-name=sah-outer
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
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
export ROUND_ID="${ROUND_ID:-1}"
export TASKS="${TASKS:-eft__math__circle_packing}"
export K="${K:-8}"
export MAX_EVALS="${MAX_EVALS:-20}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-180}"
# A per-evaluator timeout cannot close a trajectory stuck in a model/tool call.
# Keep this below the four-hour allocation while leaving enough time for the
# normal 20-evaluation inner loop and round collection.  A watchdog expiry is
# charged as launched; any score already written to its checkpoint remains an
# interruption fallback, but the candidate cannot consume more wall time.
export ROLLOUT_WALL_TIMEOUT="${ROLLOUT_WALL_TIMEOUT:-10800}"
export BASES_FILE="${BASES_FILE:-}"
export MPHI_PATH="${MPHI_PATH:-}"   # merged M_phi ckpt (step>=2): replica 0 = proposer, rest = frozen M0
export ROLLOUT_PAR="${ROLLOUT_PAR:-8}"
export PROPOSE_PAR="${PROPOSE_PAR:-8}"
export PROPOSER_SEED="${PROPOSER_SEED:-$((1000 + ROUND_ID))}"
export LOGICAL_ROUND_INDEX="${LOGICAL_ROUND_INDEX:-}"
export H2_SEED_BASE="${H2_SEED_BASE:-200000}"
export SAH_FIXED_INFERENCE_SLOTS="${SAH_FIXED_INFERENCE_SLOTS:-0}"
export SAH_ANALYSIS_REQUIRED="${SAH_ANALYSIS_REQUIRED:-0}"

export OUT_DIR="$RUN_ROOT/self_adapt_harness/outer${OUT_TAG:+-$OUT_TAG}"
mkdir -p "$OUT_DIR"
ROUND_DIR="$OUT_DIR/round$(printf '%03d' "$ROUND_ID")"
# Completed rounds are immutable evidence.  A stale/requeued controller once
# resubmitted SQL round900 and replaced its proposal metadata while the original
# summary still existed.  Refuse before starting the container so a duplicate
# cannot consume GPUs or clobber any completed-round artifact.
if [ -s "$ROUND_DIR/round_summary.json" ]; then
  echo "refusing to overwrite completed outer round: $ROUND_DIR" >&2
  exit 42
fi

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --export=ALL \
  --container-image="$IMAGE" \
  --container-mounts="/lustre:/lustre,/tmp:/tmp" \
  --container-workdir="$REPO" \
  bash "$REPO/scripts/_outer_round_worker.sh"

echo "round dir: $OUT_DIR/round$(printf '%03d' "$ROUND_ID")"
