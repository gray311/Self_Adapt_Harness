#!/usr/bin/env bash
# Runs INSIDE the aarch64 container (launched by serve_and_run_baseline.sbatch).
# Installs deps, serves N vLLM replicas of M0, shards the task list across them,
# runs the H2 baseline in parallel, and merges the per-shard summaries.
# Inputs via env: OUT_DIR N_REPLICAS MAX_EVALS TASK_IDS MODEL_PATH SERVED_MODEL VLLM_ENV REPO
set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
log(){ echo "[$(date -Is)] $*"; }

# --- evaluator deps + pinned Cordis/DSH runtime ---------------------------- #
export UV_BREAK_SYSTEM_PACKAGES=1
log "installing deps"
uv pip install --system jax optax orjson cvxpy > "$OUT_DIR/pip.log" 2>&1 \
  || { tail -30 "$OUT_DIR/pip.log"; exit 1; }
"$REPO/cordis/bootstrap.sh" >/dev/null
python3 -c "import jax, optax, orjson, cvxpy; print('evaluator deps OK')"

# --- serve N replicas (one per GPU, ports 8800+g) --- #
export VLLM_USE_FLASHINFER_SAMPLER=0
declare -a VPIDS=()
for g in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((8800 + g))
  CUDA_VISIBLE_DEVICES=$g "$VLLM_ENV/bin/python" "$VLLM_ENV/bin/vllm" serve "$MODEL_PATH" \
    --host 0.0.0.0 --port "$port" --served-model-name "$SERVED_MODEL" \
    --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 --enforce-eager --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > "$OUT_DIR/vllm-$g.log" 2>&1 &
  VPIDS+=($!)
done
trap 'for p in "${VPIDS[@]:-}"; do kill -KILL "$p" 2>/dev/null || true; done' EXIT

log "waiting for $N_REPLICAS replica(s) to become ready"
for g in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((8800 + g)); ok=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "${VPIDS[$g]}" 2>/dev/null || { log "replica $g died"; tail -40 "$OUT_DIR/vllm-$g.log"; exit 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { log "replica $g not ready in time"; exit 1; }
  log "replica $g ready (:$port)"
done

# --- shard tasks round-robin across replicas, run in parallel --- #
read -ra TASKS <<< "$TASK_IDS"
export OPENAI_API_KEY=EMPTY
cd "$REPO/src"
declare -a RPIDS=()
for g in $(seq 0 $((N_REPLICAS - 1))); do
  shard=()
  for i in "${!TASKS[@]}"; do [ $((i % N_REPLICAS)) = "$g" ] && shard+=("${TASKS[$i]}"); done
  [ ${#shard[@]} = 0 ] && continue
  port=$((8800 + g))
  log "shard $g (:$port) -> ${shard[*]}"
  OPENAI_BASE_URL="http://127.0.0.1:$port/v1" python3 -m inner.cli.run_baseline \
    --ids "${shard[@]}" --base-url "http://127.0.0.1:$port/v1" --model "$SERVED_MODEL" \
    --max-evals "$MAX_EVALS" ${EVAL_TIMEOUT:+--eval-timeout "$EVAL_TIMEOUT"} \
    --eval-python python3 --require-trajectory --out "$OUT_DIR/shard-$g" \
    > "$OUT_DIR/run-$g.log" 2>&1 &
  RPIDS+=($!)
done
rc=0
for p in "${RPIDS[@]}"; do wait "$p" || rc=1; done
log "all shards finished (rc=$rc)"
[ "$rc" -eq 0 ] || { log "one or more shards failed; refusing to merge"; exit "$rc"; }
python3 "$REPO/scripts/runtime/audit_trajectories.py" "$OUT_DIR"

# --- merge per-shard summaries --- #
python3 - "$OUT_DIR" <<'PY'
import json, glob, csv, sys
out = sys.argv[1]
rows = []
for f in glob.glob(f"{out}/shard-*/*/summary.json"):
    try: rows += json.load(open(f))
    except Exception: pass
rows.sort(key=lambda r: r.get("task_id", ""))
json.dump(rows, open(f"{out}/summary_all.json", "w"), indent=2)
cols = ["task_id", "source", "mode", "seed_score", "best_score", "delta", "evaluations", "error"]
with open(f"{out}/summary_all.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows: w.writerow(r)
imp = sum(1 for r in rows if (r.get("delta") or 0) > 1e-9)
print(f"MERGED {len(rows)} tasks | improved over seed: {imp}")
for r in rows:
    print(f"  {r.get('task_id'):40} seed={r.get('seed_score')} best={r.get('best_score')} d={r.get('delta')} err={r.get('error')}")
PY
log "done -> $OUT_DIR/summary_all.csv"
