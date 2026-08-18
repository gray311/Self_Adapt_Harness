# src/training — GRPO updates for M_phi (the only trainable parameters)

We train the proposer LoRA `phi` with **Weave_v2's proven offline-GRPO stack**
(slime FSDP backend, already running on this cluster for Qwen3.5-9B LoRA):
`Weave_v2/actions/grpo` + `scripts/train/run_qwen35_grpo_offline_lora.sh` +
`scripts/train/train_qwen35_lora.slurm` + `scripts/merge/merge.slurm`.
M0 (the executor) is never touched (plan.md §0).

## Pipeline per round

```
outer round r  ->  round_dir/grpo_batch.jsonl          (prompt, response, reward, ADVANTAGE)
      |
      v  python src/training/grpo_to_replay.py --rounds <round_dirs> --out replay.jsonl
replay.jsonl   ->  Weave slime replay format           (messages + metadata.advantage)
      |
      v  sbatch Weave_v2/scripts/train/train_qwen35_lora.slurm   (env below)
LoRA ckpt      ->  sbatch Weave_v2/scripts/merge/merge.slurm     -> merged HF ckpt
      |
      v  next round: outer_round.sbatch MODEL_PATH=<merged ckpt>
```

## Key facts inherited from Weave's implementation

- Loss: clipped policy gradient (`slime/utils/ppo_utils.py:compute_policy_loss`,
  eps_clip 0.2), sequence-level aggregation, **no KL term** by default
  (add `--use-kl-loss --kl-loss-coef ... --kl-loss-type k3` for plan.md §9.4 KL
  regularization once rounds go multi-epoch).
- Advantages are **precomputed offline** (ours: group-normalized over the K=8
  candidates in `outer/rewards.py`); slime broadcasts them per token
  (`--advantage-estimator grpo --disable-rewards-normalization`).
- LoRA: rank 64 / alpha 128 on q,k,v,o,gate,up,down (Weave's config), FSDP on
  1 node x 4 GPUs, ~2h wall.
- Loss mask: `Weave_v2/common/qwen35_mask.py` only counts assistant turns that
  are **closed by a later user/tool message** — our converter appends a
  terminal `{"role":"user","content":"ok"}` turn for exactly this reason.

## Launch (round r)

```bash
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
W=$CODE_ROOT/Weave_v2
SAH=$CODE_ROOT/self_adapt_harness
OUT=$RUN_ROOT/self_adapt_harness/outer

# 1) convert round batches (can accumulate several rounds)
python3 $SAH/src/training/grpo_to_replay.py \
    --rounds $OUT/round001 --out $OUT/grpo/round001/replay.jsonl

# 2) train phi (reuse Weave's slurm + launcher verbatim)
cd $W && \
RUN_SCRIPT=$W/scripts/train/run_qwen35_grpo_offline_lora.sh \
PROMPT_DATA=$OUT/grpo/round001/replay.jsonl \
SAVE_CKPT=$MODEL_ROOT/checkpoints/self_adapt_harness/mphi_r001 \
HF_CKPT=$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
LORA_RANK=64 LORA_ALPHA=128 NUM_EPOCH=3 GLOBAL_BATCH_SIZE=8 MICRO_BATCH_SIZE=1 \
sbatch scripts/train/train_qwen35_lora.slurm
# round r>1 continual: add LOAD_CKPT=<prev SAVE_CKPT> LORA_RESUME=1

# 3) merge LoRA -> HF ckpt vLLM can serve
MERGE_SCRIPT=$W/scripts/merge/merge_in_container.sh \
HF_CKPT=$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
CKPT_DIR=$MODEL_ROOT/checkpoints/self_adapt_harness/mphi_r001 \
OUT=$MODEL_ROOT/exports/self_adapt_harness/mphi_r001 \
sbatch scripts/merge/merge.slurm

# 4) next outer round with the trained proposer
ROUND_ID=2 BASE_HARNESS=$OUT/round001/cand<BEST> \
MODEL_PATH=$MODEL_ROOT/exports/self_adapt_harness/mphi_r001 \
sbatch $SAH/scripts/outer_round.sbatch
```

**Note on serving in round r>1:** `outer_round.sbatch` serves ONE checkpoint for
both roles. Serving the merged M_phi as the executor too would violate the
frozen-M0 rule — so once phi != 0, serve two models (replica 0 = merged M_phi
for propose; replicas 1..3 = frozen base for the inner loop) or use vLLM
multi-LoRA with the executor route pinned to no-adapter (plan.md §6.3). The
worker script's propose step already targets :8800 only; splitting the serve
block is a ~5-line change flagged with TODO(round2) in `_outer_round_worker.sh`.

## Caveats

- `grpo_batch.jsonl` rows with |advantage| < 1e-6 are dropped by the converter
  (zero-variance groups carry no gradient), matching Weave's grpo_prep.
- 8 rows/round is a small batch; accumulate 2-4 rounds per GRPO update
  (`--rounds round001 round002 ...`) or raise K if the signal is too noisy.
- Weave launcher prerequisites (their repo, already satisfied there): 6
  openclaw slime patches + `weave-train-pydeps` (aarch64) + `tfpatch`.
