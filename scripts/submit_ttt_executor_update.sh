#!/usr/bin/env bash
# Submit one executor-LoRA update and dependent merge FROM THE LOGIN HOST.
# Usage: submit_ttt_executor_update.sh TASK_TAG STEP REPLAY_JSONL [PREV_LORA_ROOT]
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

TAG="${1:?task tag}"; STEP="${2:?step}"; DATA="${3:?replay jsonl}"
PREV="${4:-}"
PREFIX="${TTT_PREFIX:-ttt12b}"
W="$CODE_ROOT/Weave_v2"
BASE="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SAVE="$MODEL_ROOT/checkpoints/self_adapt_harness/${PREFIX}_${TAG}_u${STEP}"
MERGED="$MODEL_ROOT/exports/self_adapt_harness/${PREFIX}_${TAG}_u${STEP}"
IMAGE="/lustre/fsw/portfolios/av/users/yingzim/sqsh/dgemma-core-aarch64.sqsh"
N=$(wc -l < "$DATA")
[ "$N" -ge 8 ] || {
  echo "executor update needs >=8 distinct replay rows, got $N" >&2
  exit 2
}
[ $((N % 4)) -eq 0 ] || { echo "replay rows must be divisible by 4, got $N" >&2; exit 2; }
mkdir -p "$SAVE"

# The inference-16 comparison matches the proposer update capacity and
# optimizer cadence.  Keep legacy TTT-style recovery runs available behind the
# default branch so historical lineages remain reproducible.
MATCH_PROPOSER="${TTT_MATCH_PROPOSER_TRAIN:-0}"
if [ "$MATCH_PROPOSER" = 1 ]; then
  [ "$N" -le 16 ] || {
    echo "inference-16 executor replay cannot exceed 16 rows, got $N" >&2
    exit 2
  }
  TTT_KL="${TTT_KL_COEF:-0.05}"
  TTT_LEARNING_RATE="${TTT_LR:-3e-5}"
  TTT_LORA_RANK=64
  TTT_LORA_ALPHA=128
  TTT_EPOCHS=3
  # One boundary per epoch, matching proposer K=8/GBS=8.  If a launched H2
  # trajectory fails, prepare.py truncates to a four-way-safe valid subset and
  # GBS follows that subset, preserving one boundary per epoch without
  # pretending the failed trajectory was usable training data.
  TTT_GLOBAL_BATCH_SIZE="$N"
else
  TTT_LORA_RANK=32
  TTT_LORA_ALPHA=64
  TTT_EPOCHS=1
  TTT_GLOBAL_BATCH_SIZE=4

  # Legacy budget-scaled reference settings retained for old recovery scripts.
  if [ -n "${TTT_KL_COEF:-}" ]; then
    TTT_KL="$TTT_KL_COEF"
  elif [[ "$TAG" == ahc* ]]; then
    TTT_KL=0.01
  else
    TTT_KL=0.1
  fi

  if [ -n "${TTT_LR:-}" ]; then
    TTT_LEARNING_RATE="$TTT_LR"
  elif [ "$TAG" = "ahc058" ]; then
    TTT_LEARNING_RATE=2e-5
  else
    TTT_LEARNING_RATE=4e-5
  fi
fi

train_env=(
  RUN_SCRIPT="$W/scripts/train/run_qwen35_grpo_offline_lora.sh"
  PROMPT_DATA="$DATA" SAVE_CKPT="$SAVE" HF_CKPT="$BASE"
  LR="$TTT_LEARNING_RATE" KL_COEF="$TTT_KL"
  LORA_RANK="$TTT_LORA_RANK" LORA_ALPHA="$TTT_LORA_ALPHA" NUM_GPUS=4
  NUM_EPOCH="$TTT_EPOCHS" ROLLOUT_BATCH_SIZE="$N"
  GLOBAL_BATCH_SIZE="$TTT_GLOBAL_BATCH_SIZE"
  MICRO_BATCH_SIZE=1 LOG_PROBS_CHUNK_SIZE=2048 CONTAINER_IMAGE="$IMAGE"
)
if [ -n "$PREV" ]; then
  train_env+=(LOAD_CKPT="$PREV" LORA_RESUME=1)
fi

# Slurm's controller probe is occasionally stale even while sbatch works.  Use
# the operation itself as the health check and retry a bounded number of times.
# Train and merge are retried separately so a transient merge submission does
# not create a duplicate training job.
submit_train() {
  (cd "$W" && env "${train_env[@]}" sbatch --parsable --qos=short --time=02:00:00 \
    scripts/train/train_qwen35_lora.slurm)
}
submit_merge() {
  (cd "$W" && env MERGE_SCRIPT="$W/scripts/merge/merge_in_container.sh" \
    HF_CKPT="$BASE" CKPT_DIR="$SAVE" OUT="$MERGED" CONTAINER_IMAGE="$IMAGE" \
    sbatch --parsable --qos=short --time=02:00:00 --dependency="afterok:$TJ" \
    scripts/merge/merge.slurm)
}
retry_submit() {
  local fn="$1" label="$2" attempt out
  for attempt in $(seq 1 12); do
    if out=$("$fn") && [[ "$out" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$out"
      return 0
    fi
    echo "$label submission attempt $attempt failed; retrying in 15s" >&2
    sleep 15
  done
  echo "$label submission failed after 12 attempts" >&2
  return 1
}
TJ=$(retry_submit submit_train train)
MJ=$(retry_submit submit_merge merge)
printf '%s\n' "tag=$TAG step=$STEP rows=$N lr=$TTT_LEARNING_RATE kl_coef=$TTT_KL train_job=$TJ merge_job=$MJ" \
  "lora_rank=$TTT_LORA_RANK lora_alpha=$TTT_LORA_ALPHA epochs=$TTT_EPOCHS global_batch_size=$TTT_GLOBAL_BATCH_SIZE match_proposer=$MATCH_PROPOSER" \
  "lora=$SAVE" "merged=$MERGED"
