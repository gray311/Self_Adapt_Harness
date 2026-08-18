#!/usr/bin/env bash
# Runs INSIDE the dgemma container (Cordis + evaluator deps).
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH/cordis" ] || SAH="$CODE_ROOT/self_adapt_harness"
W="$CODE_ROOT/Weave_v2"
export VLLM_ENV="${VLLM_ENV:-$ENV_ROOT/weave-qwen35-vllm/0.17.1}"
BASE_HF="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
log(){ echo "[$(date -Is)] [ttt-iter] $*"; }
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$OUT_DIR"
# Install evaluator dependencies and verify the pinned Cordis runtime.
export UV_BREAK_SYSTEM_PACKAGES=1
log "installing deps"
uv pip install --system jax optax orjson cvxpy \
  > "$OUT_DIR/pip.log" 2>&1 || { tail -30 "$OUT_DIR/pip.log"; exit 1; }
"$SAH/cordis/bootstrap.sh" >/dev/null
python3 -c "import jax, optax, orjson, cvxpy, openai; print('  deps OK')" || exit 1

CKPT="$BASE_HF"; CUM=0
: > "$OUT_DIR/curve.jsonl"

for r in $(seq 1 "$ROUNDS"); do
  RD="$OUT_DIR/r$r"; mkdir -p "$RD"
  log "round $r/$ROUNDS: serving $(basename "$CKPT")"
  CUDA_VISIBLE_DEVICES=0,1,2,3 setsid "$VLLM_ENV/bin/python" "$VLLM_ENV/bin/vllm" serve "$CKPT" \
    --host 0.0.0.0 --port 8800 --served-model-name "$SERVED_MODEL" --tensor-parallel-size 4 \
    --max-model-len 131072 --max-num-seqs 24 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 --enforce-eager --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$RD/vllm.log" 2>&1 &
  VP=$!
  ok=0; for _ in $(seq 1 240); do
    curl -sf http://127.0.0.1:8800/v1/models >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$VP" 2>/dev/null || break; sleep 5; done
  [ "$ok" = 1 ] || { log "vllm failed"; tail -30 "$RD/vllm.log"; exit 1; }

  log "  generating $K solutions with the current executor"
  cd "$SAH/src"
  RPIDS=()
  for k in $(seq 0 $((K-1))); do
    OPENAI_BASE_URL="http://127.0.0.1:8800/v1" python3 -m inner.cli.run_baseline \
      --harness-dir "$SAH/src/inner/harness" --ids "$TASK" \
      --base-url "http://127.0.0.1:8800/v1" --model "$SERVED_MODEL" \
      --max-evals "$MAX_EVALS" --eval-timeout "$EVAL_TIMEOUT" \
      --temperature "$TTT_TEMP" \
      --eval-python python3 --require-trajectory --out "$RD/k$k" > "$RD/k$k.log" 2>&1 &
    RPIDS+=($!)
    # throttle on the ROLLOUT pids only -- counting all jobs would include the
    # vllm server and stall the loop
    while :; do
      run=0; for q in "${RPIDS[@]}"; do kill -0 "$q" 2>/dev/null && run=$((run+1)); done
      [ "$run" -lt 24 ] && break; sleep 10
    done
  done
  # bare `wait` also waits on the vllm server, which never exits -- that hung the
  # previous runs for hours after the rollouts had already finished
  rollout_rc=0
  for q in "${RPIDS[@]}"; do wait "$q" 2>/dev/null || rollout_rc=1; done
  kill -9 -- "-$VP" 2>/dev/null || kill -9 "$VP" 2>/dev/null || true
  sleep 20
  [ "$rollout_rc" -eq 0 ] || { log "one or more executor trajectories failed"; exit 1; }
  python3 "$SAH/scripts/runtime/audit_trajectories.py" "$RD" \
    || { log "trajectory artifact audit failed"; exit 1; }

  CUM=$((CUM + K))
  python3 - "$OUT_DIR" "$RD" "$TASK" "$CUM" "$r" <<'PY'
import json, glob, os, sys
out, rd, task, cum, r = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
pool = os.path.join(out, "pool.jsonl")
new = 0
with open(pool, "a") as ph:
    for f in glob.glob(os.path.join(rd, "k*", "**", "summary.json"), recursive=True):
        try: e = json.load(open(f))
        except Exception: continue
        e = e[0] if isinstance(e, list) else e
        s, p = e.get("best_score"), e.get("best_program")
        if s is None or not p or s <= 0: continue
        ph.write(json.dumps({"score": s, "program": p, "round": r}) + "\n"); new += 1
rows = [json.loads(l) for l in open(pool)] if os.path.exists(pool) else []
best = max((x["score"] for x in rows), default=None)
with open(os.path.join(out, "curve.jsonl"), "a") as ch:
    ch.write(json.dumps({"round": r, "cum_rollouts": cum, "best": best, "n_pool": len(rows)}) + "\n")
print(f"  round {r}: +{new} solutions, pool={len(rows)}, best={best}, cum_rollouts={cum}")
PY

  [ "$r" -lt "$ROUNDS" ] || break
  log "  training the executor on its own best solutions"
  DAT="$RD/train.jsonl"
  python3 - "$OUT_DIR/pool.jsonl" "$DAT" "$SAH" "$TASK" <<'PY'
import json, sys, yaml
pool, dst, sah, task = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
rows = [json.loads(l) for l in open(pool)]
cordis = yaml.safe_load(open(f"{sah}/src/inner/harness/cordis.yml")) or []
system_rows = [row for row in cordis if isinstance(row, dict)
               and row.get("id") == "system-prompt"]
if len(system_rows) != 1:
    raise SystemExit("inner Cordis package needs exactly one system-prompt")
system = str((system_rows[0].get("config") or {}).get("persona") or "").strip()
TOOLS = [{"type": "function", "function": {"name": "edit_solution",
  "description": "Change the code inside the # EVOLVE-BLOCK region, then call evaluate_solution to score it.",
  "parameters": {"type": "object", "properties": {"code": {"type": "string",
    "description": "SEARCH/REPLACE diff block(s), or the full replacement body."}}, "required": ["code"]}}},
 {"type": "function", "function": {"name": "evaluate_solution",
  "description": "Score the current program against the task evaluator.",
  "parameters": {"type": "object", "properties": {}, "required": []}}}]
n = len(rows); tot = sum(x["score"] for x in rows)
with open(dst, "w") as fh:
    for x in rows:
        loo = (tot - x["score"]) / (n - 1) if n > 1 else x["score"]
        call = ("<tool_call>\n<function=edit_solution>\n<parameter=code>\n"
                f"{x['program']}\n</parameter>\n</function>\n</tool_call>")
        fh.write(json.dumps({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {task}\n\nImprove the EVOLVE-BLOCK. "
                                        "Call edit_solution with the new block, then evaluate_solution."},
            {"role": "assistant", "content": call},
            {"role": "tool", "content": f"Edit applied. evaluate_solution -> combined_score {x['score']:.6f}."}],
            "tools": TOOLS,
            "metadata": {"advantage": x["score"] - loo, "reward": x["score"],
                         "task_id": task, "valid": True, "tools": TOOLS}}) + "\n")
print(f"  built {n} training rows")
PY
  N=$(wc -l < "$DAT"); [ "$N" -ge 2 ] || { log "  too few rows, stopping"; break; }
  SAVE="$MODEL_ROOT/checkpoints/self_adapt_harness/ttt_iter_$(basename "$OUT_DIR")_r$r"
  MERGED="$MODEL_ROOT/exports/self_adapt_harness/ttt_iter_$(basename "$OUT_DIR")_r$r"
  mkdir -p "$SAVE"
  TJ=$(cd "$W" && env RUN_SCRIPT="$W/scripts/train/run_qwen35_grpo_offline_lora.sh" \
    PROMPT_DATA="$DAT" SAVE_CKPT="$SAVE" HF_CKPT="$BASE_HF" LR=4e-5 KL_COEF=0.1 \
    LORA_RANK=32 LORA_ALPHA=64 NUM_GPUS=4 NUM_EPOCH=2 ROLLOUT_BATCH_SIZE="$N" \
    GLOBAL_BATCH_SIZE=8 MICRO_BATCH_SIZE=1 LOG_PROBS_CHUNK_SIZE=2048 \
    sbatch --parsable scripts/train/train_qwen35_lora.slurm 2>&1 | grep -oE '[0-9]{6,}' | tail -1)
  MJ=$(cd "$W" && env MERGE_SCRIPT="$W/scripts/merge/merge_in_container.sh" \
    HF_CKPT="$BASE_HF" CKPT_DIR="$SAVE" OUT="$MERGED" \
    sbatch --parsable --dependency=afterok:"$TJ" scripts/merge/merge.slurm 2>&1 | grep -oE '[0-9]{6,}' | tail -1)
  log "  train $TJ / merge $MJ -> waiting"
  for _ in $(seq 1 180); do
    st=$(sacct -j "$MJ" -X -n -o State 2>/dev/null | head -1 | xargs)
    case "$st" in COMPLETED) break;; FAILED|CANCELLED*|TIMEOUT) log "  train/merge failed ($st)"; break 2;; esac
    sleep 60
  done
  [ -f "$MERGED/config.json" ] && CKPT="$MERGED" || { log "  merge missing, keeping previous ckpt"; }
done
log "done; curve at $OUT_DIR/curve.jsonl"
