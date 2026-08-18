#!/usr/bin/env bash
# Runs inside the Slurm container on the allocated GPU node.
set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

REPO="/lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/yingzim/code/sah_corids"
PORT="${SAH_CORDIS_PORT:-8800}"
MODEL_PATH="${SAH_MODEL_PATH}"
SERVED_MODEL="${SAH_SERVED_MODEL:-qwen3.5-9b}"
OUT_DIR="$REPO/.runtime/cordis/real-model-${SLURM_JOB_ID:-manual}"
FLASHINFER_WORKSPACE_BASE="$REPO/.runtime/cordis/flashinfer-workspace"
mkdir -p "$OUT_DIR"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE
CUDA_VISIBLE_DEVICES=0 "$VLLM_ENV/bin/python" "$VLLM_ENV/bin/vllm" serve "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_MODEL" \
  --max-model-len 32768 --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 --enforce-eager --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  > "$OUT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
cleanup() {
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited during startup" >&2
    tail -n 100 "$OUT_DIR/vllm.log" >&2
    exit 1
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "vLLM did not become ready within 20 minutes" >&2
  tail -n 100 "$OUT_DIR/vllm.log" >&2
  exit 1
fi

export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="EMPTY"
export SAH_CORDIS_MODEL="$SERVED_MODEL"
export SAH_CORDIS_CONTEXT_WINDOW="32768"
export SAH_CORDIS_MAX_TOKENS="256"
export DSH_HOME="$OUT_DIR/dsh-home"
export SAH_CORDIS_TRAJECTORY_ROOT="$DSH_HOME/trajectories"
export DSH_TELEMETRY_DISABLED=1

"$REPO/cordis/real_smoke.sh" | tee "$OUT_DIR/result.txt"
python3 "$REPO/cordis/export_trajectory.py" \
  "$SAH_CORDIS_TRAJECTORY_ROOT" "$OUT_DIR/trajectory.jsonl" \
  --manifest "$OUT_DIR/trajectory.manifest.json" | tee -a "$OUT_DIR/result.txt"
echo "Cordis real-model artifacts: $OUT_DIR"
