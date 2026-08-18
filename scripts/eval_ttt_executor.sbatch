#!/usr/bin/env bash
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=03:30:00
#SBATCH --job-name=ttt-eval
#SBATCH --output=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.err
#
# Evaluate the "update the executor" (TTT) arm.
#
# The executor is the LoRA-tuned checkpoint; the harness is the FIXED initial
# harness and no proposer runs at all, so the only thing that differs from the
# baseline is the executor's weights. K independent rollouts, best-of-K reported.
#
# Env: TTT_CKPT (merged HF dir)  TASK  OUT_DIR  [K=8] [MAX_EVALS=20] [EVAL_TIMEOUT=240]
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
export VLLM_ENV="${VLLM_ENV:-$ENV_ROOT/weave-qwen35-vllm/0.17.1}"
: "${TTT_CKPT:?set TTT_CKPT}"; : "${TASK:?set TASK}"; : "${OUT_DIR:?set OUT_DIR}"
K="${K:-8}"; MAX_EVALS="${MAX_EVALS:-20}"; EVAL_TIMEOUT="${EVAL_TIMEOUT:-240}"
SERVED_MODEL="qwen3.5-9b"
mkdir -p "$OUT_DIR"
log(){ echo "[$(date -Is)] [ttt-eval] $*"; }

"$SAH/cordis/bootstrap.sh" >/dev/null

case "$TASK" in *ahc*) export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS=12 \
  AHC_CACHE_DIR="$SAH/ahc_work/cache";; esac

export VLLM_USE_FLASHINFER_SAMPLER=0
log "serving TTT executor: $TTT_CKPT"
CUDA_VISIBLE_DEVICES=0,1,2,3 setsid "$VLLM_ENV/bin/python" "$VLLM_ENV/bin/vllm" serve "$TTT_CKPT" \
  --host 0.0.0.0 --port 8800 --served-model-name "$SERVED_MODEL" \
  --tensor-parallel-size 4 \
  --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 --enforce-eager --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  > "$OUT_DIR/vllm.log" 2>&1 &
VP=$!
trap 'kill -9 -- "-$VP" 2>/dev/null || kill -9 "$VP" 2>/dev/null || true' EXIT
ok=0
for _ in $(seq 1 240); do
  curl -sf http://127.0.0.1:8800/v1/models >/dev/null 2>&1 && { ok=1; break; }
  kill -0 "$VP" 2>/dev/null || { log "vllm died"; tail -40 "$OUT_DIR/vllm.log"; exit 1; }
  sleep 5
done
[ "$ok" = 1 ] || { log "vllm never came up"; exit 1; }
log "vllm ready"

cd "$SAH/src"
pids=()
for k in $(seq 0 $((K-1))); do
  OPENAI_BASE_URL="http://127.0.0.1:8800/v1" python3 -m inner.cli.run_baseline \
    --harness-dir "$SAH/src/inner/harness" --ids "$TASK" \
    --base-url "http://127.0.0.1:8800/v1" --model "$SERVED_MODEL" \
    --max-evals "$MAX_EVALS" --eval-timeout "$EVAL_TIMEOUT" \
    --eval-python python3 --require-trajectory \
    --out "$OUT_DIR/k$k" > "$OUT_DIR/k$k.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
[ "$rc" -eq 0 ] || { log "one or more executor trajectories failed"; exit 1; }
python3 "$SAH/scripts/runtime/audit_trajectories.py" "$OUT_DIR" \
  || { log "trajectory artifact audit failed"; exit 1; }
log "rollouts done"

python3 - "$OUT_DIR" "$TASK" <<'PY'
import json, glob, os, sys
out, task = sys.argv[1], sys.argv[2]
best, scores = None, []
for f in glob.glob(os.path.join(out, "k*", "**", "summary.json"), recursive=True):
    try: e = json.load(open(f))
    except Exception: continue
    e = e[0] if isinstance(e, list) else e
    s = e.get("best_score")
    if s is None: continue
    scores.append(s)
    best = s if best is None else max(best, s)
json.dump({"task": task, "arm": "ttt_executor", "k": len(scores),
           "scores": scores, "best": best},
          open(os.path.join(out, "ttt_result.json"), "w"), indent=1)
print(f"  {task}: K={len(scores)} best={best} scores={[round(x,5) for x in scores]}")
PY
log "done"
