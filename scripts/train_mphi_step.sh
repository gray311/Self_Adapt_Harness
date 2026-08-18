#!/usr/bin/env bash
# Train phi on one outer step's GRPO group, then merge for serving.
#   bash scripts/train_mphi_step.sh <round_dir> <step_tag> [prev_ckpt_dir]
# e.g. bash scripts/train_mphi_step.sh $RUN_ROOT/self_adapt_harness/outer/round001 s001
#      bash scripts/train_mphi_step.sh .../round002 s002 $MODEL_ROOT/checkpoints/self_adapt_harness/mphi_s001
#
# Chain: grpo_to_replay (login node, pure python) -> Weave train_qwen35_lora.slurm
# (slime offline GRPO, LoRA r64/a128 on 4 GPUs) -> merge.slurm (afterok) ->
# merged HF ckpt at $MODEL_ROOT/exports/self_adapt_harness/mphi_<step>.
# Only phi trains; M0 is never touched (plan.md §0).
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

ROUND_DIR="${1:?usage: train_mphi_step.sh <round_dir> <step_tag> [prev_ckpt]}"
STEP="${2:?step tag, e.g. s001}"
PREV_CKPT="${3:-}"

sbatch_retry() {  # transient slurmctld failures are common under load
  local out
  for _ in $(seq 1 30); do
    if out=$("$@" 2>&1); then echo "$out" | tail -1; return 0; fi
    echo "  (sbatch failed: $(echo "$out" | tail -1); retrying in 60s)" >&2
    sleep 60
  done
  echo "sbatch failed after 30 attempts" >&2; return 1
}

W="$CODE_ROOT/Weave_v2"
SAH="$CODE_ROOT/self_adapt_harness"
BASE_HF="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SAVE_CKPT="$MODEL_ROOT/checkpoints/self_adapt_harness/mphi_$STEP"
MERGED="$MODEL_ROOT/exports/self_adapt_harness/mphi_$STEP"
REPLAY="$ROUND_DIR/replay.jsonl"
REPLAY_MANIFEST="$ROUND_DIR/replay_manifest.json"

echo "[1/3] convert grpo_batch -> slime replay"
python3 "$SAH/src/training/grpo_to_replay.py" --rounds "$ROUND_DIR" --out "$REPLAY"
N_ROWS=$(wc -l < "$REPLAY")
GENERATED_ROWS="$N_ROWS"
MIN_TRAINABLE_ROWS="${MIN_TRAINABLE_ROWS:-4}"
[ "$N_ROWS" -ge "$MIN_TRAINABLE_ROWS" ] || {
  python3 - "$REPLAY_MANIFEST" "$GENERATED_ROWS" "$MIN_TRAINABLE_ROWS" <<'PY'
import json, os, sys
path, generated, minimum = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
payload = {
    "schema": "h1-replay/1.0", "status": "skipped_below_minimum",
    "generated_trainable_rows": generated, "minimum_trainable_rows": minimum,
    "archive_mixed_rows": 0, "zero_advantage_padding_rows": 0,
    "optimizer_rows": generated,
}
tmp = path + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, path)
PY
  echo "only $N_ROWS trainable rows (<$MIN_TRAINABLE_ROWS) — protocol skips this update"
  exit 4
}

# Cross-step archive: bank this round's strongly-positive rows, and (opt-in
# via ARCHIVE_MIX=n) mix n archived winner-trajectories from OTHER tasks into
# the replay — amortization signal + small-batch noise damping. Off by default.
ARCHIVE="$(dirname "$ROUND_DIR")/replay_archive.jsonl"
python3 - "$REPLAY" "$ARCHIVE" "${ARCHIVE_MIX:-0}" <<'PYEOF'
import json, sys
replay, archive, mix = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(replay)]
# bank winners (advantage > 0.3) into the archive, dedup by (task, round, k)
try:
    arch = [json.loads(l) for l in open(archive)]
except FileNotFoundError:
    arch = []
seen = {(a["metadata"]["task_id"], a["metadata"]["round"], a["metadata"]["k"]) for a in arch}
for r in rows:
    md = r["metadata"]
    if md.get("advantage", 0) > 0.3 and (md["task_id"], md["round"], md["k"]) not in seen:
        arch.append(r)
open(archive, "w").write("".join(json.dumps(a) + "\n" for a in arch))
if mix > 0 and arch:
    cur_tasks = {r["metadata"]["task_id"] for r in rows}
    pool = [a for a in arch if a["metadata"]["task_id"] not in cur_tasks]
    pool.sort(key=lambda a: -a["metadata"]["round"])  # freshest first
    extra = pool[:mix]
    if extra:
        open(replay, "a").write("".join(json.dumps(a) + "\n" for a in extra))
        print(f"[archive] mixed {len(extra)} winner rows from other tasks "
              f"(archive size {len(arch)})")
print(f"[archive] banked; archive={len(arch)} rows")
PYEOF
N_ROWS=$(wc -l < "$REPLAY")
PREPAD_ROWS="$N_ROWS"

# Pad short groups (e.g. after sanitize_grpo_batch drops poisoned rows) up to
# GLOBAL_BATCH_SIZE with zero-advantage copies: zero advantage => zero policy-
# gradient contribution, they only satisfy slime's fixed batch geometry.
GBS=8
if [ "$N_ROWS" -lt "$GBS" ]; then
  python3 - "$REPLAY" "$GBS" <<'PY'
import json, sys
path, gbs = sys.argv[1], int(sys.argv[2])
rows = [json.loads(l) for l in open(path)]
i = 0
while len(rows) < gbs:
    pad = json.loads(json.dumps(rows[i % len(rows)]))
    if isinstance(pad.get("metadata"), dict) and "advantage" in pad["metadata"]:
        pad["metadata"]["advantage"] = 0.0
    if "advantage" in pad:
        pad["advantage"] = 0.0
    rows.append(pad); i += 1
open(path, "w").write("".join(json.dumps(r) + "\n" for r in rows))
print(f"[pad] replay padded to {gbs} rows (zero-advantage fillers)")
PY
  N_ROWS="$GBS"
fi

python3 - "$REPLAY_MANIFEST" "$GENERATED_ROWS" "$PREPAD_ROWS" "$N_ROWS" \
  "$MIN_TRAINABLE_ROWS" "${ARCHIVE_MIX:-0}" <<'PY'
import json, os, sys
path = sys.argv[1]
generated, prepad, optimizer, minimum, requested_mix = map(int, sys.argv[2:])
payload = {
    "schema": "h1-replay/1.0", "status": "optimizer_input_ready",
    "generated_trainable_rows": generated,
    "minimum_trainable_rows": minimum,
    "archive_mix_requested": requested_mix,
    "archive_mixed_rows": max(0, prepad - generated),
    "zero_advantage_padding_rows": max(0, optimizer - prepad),
    "optimizer_rows": optimizer,
}
tmp = path + ".tmp"
open(tmp, "w").write(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, path)
PY

echo "[2/3] submit GRPO training ($N_ROWS rows, LoRA r64/a128, lr ${LR:-3e-5}, ${NUM_EPOCH:-3} epochs)"
mkdir -p "$SAVE_CKPT" "$LOG_ROOT/slurm"
TRAIN_ENV=(RUN_SCRIPT="$W/scripts/train/run_qwen35_grpo_offline_lora.sh"
           PROMPT_DATA="$REPLAY" SAVE_CKPT="$SAVE_CKPT" HF_CKPT="$BASE_HF"
           LR="${LR:-3e-5}" KL_COEF="${KL_COEF:-0.05}" LORA_RANK=64 LORA_ALPHA=128 NUM_GPUS=4 NUM_EPOCH="${NUM_EPOCH:-3}"
           ROLLOUT_BATCH_SIZE="$N_ROWS" GLOBAL_BATCH_SIZE=8 MICRO_BATCH_SIZE=1
           LOG_PROBS_CHUNK_SIZE=2048)
[ -n "$PREV_CKPT" ] && TRAIN_ENV+=(LOAD_CKPT="$PREV_CKPT" LORA_RESUME=1)
TRAIN_JOB=$(cd "$W" && sbatch_retry env "${TRAIN_ENV[@]}" sbatch --parsable scripts/train/train_qwen35_lora.slurm)
echo "  train job: $TRAIN_JOB"

echo "[3/3] submit merge (afterok:$TRAIN_JOB)"
MERGE_JOB=$(cd "$W" && sbatch_retry env MERGE_SCRIPT="$W/scripts/merge/merge_in_container.sh" \
  HF_CKPT="$BASE_HF" CKPT_DIR="$SAVE_CKPT" OUT="$MERGED" \
  sbatch --parsable --dependency=afterok:"$TRAIN_JOB" scripts/merge/merge.slurm)
echo "  merge job: $MERGE_JOB"
echo
echo "merged M_phi -> $MERGED"
echo "next step example:"
echo "  ROUND_ID=<r+1> TASKS=<next_task> BASES_FILE=$ROUND_DIR/next_bases.json \\"
echo "  MPHI_PATH=$MERGED sbatch $SAH/scripts/outer_round.sbatch"
