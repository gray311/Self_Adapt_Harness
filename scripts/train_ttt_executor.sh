#!/usr/bin/env bash
# Train the EXECUTOR on its own high-scoring solutions -- the "update executor"
# (test-time RL) arm of the score-compute comparison.
#
#   train_ttt_executor.sh <task_id> <tag> [frac]
#
# frac (default 1.0) takes a chronological PREFIX of the self-generated rollouts,
# so a checkpoint corresponds to a point on the executor-rollout budget axis.
# No external solutions are involved: every training row is a program this same
# frozen model produced.
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
W="$CODE_ROOT/Weave_v2"; SAH="$CODE_ROOT/self_adapt_harness"
BASE_HF="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TASK="${1:?task}"; TAG="${2:?tag}"; FRAC="${3:-1.0}"
SRC="$RUN_ROOT/self_adapt_harness/ttt_arm/$TASK.jsonl"
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
DAT="$RUN_ROOT/self_adapt_harness/ttt_arm/${TAG}.train.jsonl"
python3 - "$SRC" "$DAT" "$FRAC" <<'PY'
import sys
src,dst,frac=sys.argv[1],sys.argv[2],float(sys.argv[3])
rows=open(src).read().splitlines()
n=max(2,int(len(rows)*frac))
open(dst,"w").write("\n".join(rows[:n])+"\n")
print(f"  prefix {n}/{len(rows)} rows ({frac:.0%} of the rollout budget)")
PY
N=$(wc -l < "$DAT")
SAVE="$MODEL_ROOT/checkpoints/self_adapt_harness/ttt_$TAG"
mkdir -p "$SAVE"
J=$(cd "$W" && env RUN_SCRIPT="$W/scripts/train/run_qwen35_grpo_offline_lora.sh" \
  PROMPT_DATA="$DAT" SAVE_CKPT="$SAVE" HF_CKPT="$BASE_HF" \
  LR="${LR:-3e-5}" KL_COEF="${KL_COEF:-0.05}" LORA_RANK=64 LORA_ALPHA=128 NUM_GPUS=4 \
  NUM_EPOCH="${NUM_EPOCH:-2}" ROLLOUT_BATCH_SIZE="$N" GLOBAL_BATCH_SIZE=8 MICRO_BATCH_SIZE=1 \
  LOG_PROBS_CHUNK_SIZE=2048 \
  sbatch --parsable scripts/train/train_qwen35_lora.slurm 2>&1 | grep -oE '[0-9]{6,}' | tail -1)
MERGED="$MODEL_ROOT/exports/self_adapt_harness/ttt_$TAG"
M=$(cd "$W" && env MERGE_SCRIPT="$W/scripts/merge/merge_in_container.sh" \
  HF_CKPT="$BASE_HF" CKPT_DIR="$SAVE" OUT="$MERGED" \
  sbatch --parsable --dependency=afterok:"$J" scripts/merge/merge.slurm 2>&1 | grep -oE '[0-9]{6,}' | tail -1)
echo "  ttt_$TAG: $N rows, train job $J, merge job $M -> $MERGED"
echo "$TASK $TAG $FRAC $N $J $M $MERGED" >> "$RUN_ROOT/self_adapt_harness/ttt_arm/trainings.txt"
