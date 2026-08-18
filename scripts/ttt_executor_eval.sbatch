#!/usr/bin/env bash
#SBATCH --account=av_alpamayo_reasoning
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=02:30:00
#SBATCH --job-name=ttt12-eval
#SBATCH --output=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/av/users/yingzim/logs/slurm/%x-%j.err

# Evaluate a genuinely updated executor with the immutable initial H2 harness.
# Required: TASK TTT_CKPT OUT_DIR. Optional: K, MAX_EVALS, PARENT_FILE.
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
VLLM_ENV="${VLLM_ENV:-$ENV_ROOT/weave-qwen35-vllm/0.17.1}"
IMAGE="${IMAGE:-/lustre/fsw/portfolios/av/users/yingzim/sqsh/dgemma-core-aarch64.sqsh}"
: "${TASK:?}"; : "${TTT_CKPT:?}"; : "${OUT_DIR:?}"
K="${K:-16}"; MAX_EVALS="${MAX_EVALS:-20}"; EVAL_TIMEOUT="${EVAL_TIMEOUT:-420}"
K_START="${K_START:-0}"; WRITE_MANIFEST="${WRITE_MANIFEST:-1}"
H2_SEED_BASE="${H2_SEED_BASE:-200000}"
H2_BATCH_INDEX="${H2_BATCH_INDEX:-0}"
PARENT_FILE="${PARENT_FILE:-}"
CHARGED_ROLLOUTS="${CHARGED_ROLLOUTS:-$K}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
AHC_CASE_WORKERS="${AHC_CASE_WORKERS:-}"
if [[ "$TASK" == *ahc* ]]; then
  MAX_PARALLEL="${MAX_PARALLEL:-8}"
  AHC_CASE_WORKERS="${AHC_CASE_WORKERS:-6}"
else
  MAX_PARALLEL="${MAX_PARALLEL:-24}"
fi
mkdir -p "$OUT_DIR"
if [ -n "${RUNTIME_SOURCE_MANIFEST:-}" ]; then
  python3 "$SAH/scripts/runtime/provenance.py" verify \
    --manifest "$RUNTIME_SOURCE_MANIFEST"
fi
if [ -n "${FIXED_H2_SHA256:-}" ]; then
  python3 "$SAH/scripts/runtime/hash_h2_package.py" "$SAH/src/inner/harness" \
    --expect "$FIXED_H2_SHA256" >/dev/null
fi
date +%s > "$OUT_DIR/worker_started_epoch"
# The worker is a fresh shell inside the container.  Explicitly export every
# value it consumes; plain shell assignments are otherwise not inherited.
export SAH VLLM_ENV IMAGE TASK TTT_CKPT OUT_DIR K MAX_EVALS EVAL_TIMEOUT PARENT_FILE \
  MAX_PARALLEL AHC_CASE_WORKERS CHARGED_ROLLOUTS K_START WRITE_MANIFEST
export H2_SEED_BASE
export H2_BATCH_INDEX
export FIXED_H2_SHA256

srun --nodes=1 --ntasks=1 --cpus-per-task=64 --export=ALL \
  --container-image="$IMAGE" --container-mounts="/lustre:/lustre,/tmp:/tmp" \
  --container-workdir="$SAH" bash -s <<'WORKER'
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
: "${SAH:?missing exported SAH repository path}"
export UV_BREAK_SYSTEM_PACKAGES=1 VLLM_USE_FLASHINFER_SAMPLER=0
case "$TASK" in
  *ahc*) export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS \
    AHC_CACHE_DIR="$SAH/ahc_work/cache" ;;
esac
log_suffix=""
[ "$K_START" -gt 0 ] && log_suffix="_k${K_START}"
uv pip install --system jax optax orjson cvxpy \
  > "$OUT_DIR/pip${log_suffix}.log" 2>&1
"$SAH/cordis/bootstrap.sh" >/dev/null

CUDA_VISIBLE_DEVICES=0,1,2,3 setsid "$VLLM_ENV/bin/python" "$VLLM_ENV/bin/vllm" serve "$TTT_CKPT" \
  --host 0.0.0.0 --port 8800 --served-model-name qwen3.5-9b --tensor-parallel-size 4 \
  --max-model-len 131072 --max-num-seqs 24 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 --enforce-eager --language-model-only \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml > "$OUT_DIR/vllm${log_suffix}.log" 2>&1 &
VP=$!
cleanup(){ kill -9 -- "-$VP" 2>/dev/null || kill -9 "$VP" 2>/dev/null || true; }
trap cleanup EXIT
ok=0
for _ in $(seq 1 240); do
  curl -sf http://127.0.0.1:8800/v1/models >/dev/null 2>&1 && { ok=1; break; }
  kill -0 "$VP" 2>/dev/null || break
  sleep 5
done
[ "$ok" = 1 ] || { tail -60 "$OUT_DIR/vllm${log_suffix}.log"; exit 1; }

cd "$SAH/src"
pids=()
started=0
for k in $(seq "$K_START" $((K_START+K-1))); do
  shopt -s nullglob
  completed=("$OUT_DIR/k$k"/*/summary.json)
  shopt -u nullglob
  [ "${#completed[@]}" -gt 0 ] && continue
  extra=()
  [ -n "$PARENT_FILE" ] && extra+=(--seed-programs-file "$PARENT_FILE")
  OPENAI_BASE_URL=http://127.0.0.1:8800/v1 python3 -m inner.cli.run_baseline \
    --harness-dir "$SAH/src/inner/harness" --ids "$TASK" \
    --base-url http://127.0.0.1:8800/v1 --model qwen3.5-9b \
    --max-evals "$MAX_EVALS" --eval-timeout "$EVAL_TIMEOUT" --temperature 0.7 \
    --seed "$((H2_SEED_BASE + H2_BATCH_INDEX * 16 + k))" \
    --eval-python python3 --require-trajectory "${extra[@]}" --out "$OUT_DIR/k$k" \
    > "$OUT_DIR/k$k.log" 2>&1 &
  pids+=("$!")
  started=$((started+1))
  while :; do
    active=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && active=$((active+1)); done
    [ "$active" -lt "$MAX_PARALLEL" ] && break
    sleep 5
  done
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
cleanup
trap - EXIT
python3 "$SAH/scripts/runtime/audit_trajectories.py" "$OUT_DIR"
if [ "$WRITE_MANIFEST" != 1 ]; then
  echo "supplemental batch complete: k_start=$K_START k=$K started=$started rc=$rc"
  exit "$rc"
fi
python3 - "$OUT_DIR" "$TASK" "$TTT_CKPT" "$K" "$CHARGED_ROLLOUTS" "$started" "$rc" <<'PY'
import glob, json, os, sys, time
from pathlib import Path
from inner.runtime.package_hash import h2_sha256
out, task, ckpt = sys.argv[1:4]
target, charged, attempt, rc = map(int, sys.argv[4:8])
rows=[]
for f in glob.glob(os.path.join(out,"k*","*","summary.json")):
    try: data=json.load(open(f))
    except Exception: continue
    data=data if isinstance(data,list) else [data]
    e=next((x for x in data if x.get("task_id")==task),None)
    if e and e.get("best_score") is not None:
        rows.append({"summary":f,"score":e["best_score"],"evaluations":e.get("evaluations",0),
                     "executor_model_calls":int((e.get("ledger") or {}).get("llm_calls") or 0)})
hb="/lustre/fsw/portfolios/av/users/yingzim/code/self_adapt_harness/src/inner/harness"
manifest={"schema":1,"task":task,"checkpoint":ckpt,"target":target,
          "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
          "launched":charged,"attempt_launched":attempt,"usable":len(rows),
          "evaluator_calls":sum(x["evaluations"] for x in rows),
          "executor_model_calls":sum(x["executor_model_calls"] for x in rows),
          "batch_best":max((x["score"] for x in rows),default=None),"worker_rc":rc,
          "fixed_harness_sha256":h2_sha256(Path(hb)),
          "fixed_harness_hash_scheme":"canonical-h2-v1",
          "h2_batch_index":int(os.environ.get("H2_BATCH_INDEX","0")),
          "h2_decode_seed_base":int(os.environ.get("H2_SEED_BASE","200000")),
          "h2_decode_seeds":[int(os.environ.get("H2_SEED_BASE","200000"))
             + int(os.environ.get("H2_BATCH_INDEX","0"))*16 + k
             for k in range(target)],
          "finished_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
expected=os.environ.get("FIXED_H2_SHA256")
if expected and manifest["fixed_harness_sha256"] != expected:
    raise SystemExit("fixed H2 changed during executor comparison")
json.dump(manifest,open(os.path.join(out,"eval_manifest.json"),"w"),indent=2)
print(json.dumps(manifest,indent=2))
if rc: raise SystemExit("one or more executor trajectories failed")
if len(rows)<2: raise SystemExit("fewer than two usable executor trajectories")
PY
WORKER
