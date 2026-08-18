#!/usr/bin/env bash
# Runs INSIDE the aarch64 container (launched by outer_round.sbatch).
# One full outer round:
#   1. deps + serve N vLLM replicas of the frozen base checkpoint
#      (M0 executor; M_phi proposer = same weights in round 1 / merged phi later)
#   2. propose: M_phi + H1 -> K candidate H2 packages (outer.outer_round propose)
#   3. rollouts: for each valid candidate, one process runs the inner loop over
#      the round's task list (<=MAX_EVALS evals/task), sharded across replicas
#   4. collect: rewards + GRPO advantages -> grpo_batch.jsonl
# Env in: OUT_DIR ROUND_ID K N_REPLICAS MAX_EVALS EVAL_TIMEOUT MODEL_PATH
#         SERVED_MODEL VLLM_ENV REPO BASE_HARNESS PROPOSER_SEED
set -euo pipefail
umask 027
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
log(){ echo "[$(date -Is)] $*"; }

# Paired reward is only meaningful when its counterfactual rollouts are
# actually generated.  Make the reward selector authoritative so proposal-time
# metadata and the rollout phase cannot silently disagree.
if [ "${SAH_ADV:-}" = "paired" ]; then
  export SAH_PAIRED_CONTROLS=1
fi
PAIRED_REPEATS="${SAH_PAIRED_REPEATS:-1}"
[[ "$PAIRED_REPEATS" =~ ^[0-9]+$ ]] && [ "$PAIRED_REPEATS" -ge 1 ] || {
  echo "SAH_PAIRED_REPEATS must be a positive integer" >&2
  exit 2
}
if [ "${SAH_PAIRED_CONTROLS:-0}" != 1 ]; then
  PAIRED_REPEATS=1
fi
PAIR_SEED_STRIDE=$((K * PAIRED_REPEATS))
[ "$PAIR_SEED_STRIDE" -ge 16 ] || PAIR_SEED_STRIDE=16

ROUND_DIR="$OUT_DIR/round$(printf '%03d' "$ROUND_ID")"
mkdir -p "$ROUND_DIR"
if [ -n "${RUNTIME_SOURCE_MANIFEST:-}" ]; then
  python3 "$REPO/scripts/runtime/provenance.py" verify \
    --manifest "$RUNTIME_SOURCE_MANIFEST"
fi

# --- deps --- #
export UV_BREAK_SYSTEM_PACKAGES=1
log "installing deps"
uv pip install --system -e "$CODE_ROOT/NexAU" jax optax orjson cvxpy > "$OUT_DIR/pip.log" 2>&1 \
  || { tail -30 "$OUT_DIR/pip.log"; exit 1; }
python3 -c "from nexau import Agent; import jax, optax, orjson, cvxpy, openai; print('deps OK')"

# --- serve N replicas --- #
# Split serving (plan.md §0/§6.3): when MPHI_PATH is set (step >= 2, trained
# phi), replica 0 serves the merged M_phi (PROPOSE ONLY) and replicas 1..N-1
# serve the frozen base M0 (INNER LOOP ONLY) — the executor stays bit-frozen.
# Step 1 (phi == 0): every replica serves the base and both roles share them.
# Each replica is setsid'd into its own process group so the WHOLE tree
# (APIServer + spawned EngineCore children) can be killed with kill -- -PID;
# killing only the parent leaves orphans holding GPU memory.
export VLLM_USE_FLASHINFER_SAMPLER=0
serve_replica(){ # gpu port ckpt logfile -> echoes pid (process-group leader)
  local gpu="$1" port="$2" ckpt="$3" logf="$4"
  CUDA_VISIBLE_DEVICES=$gpu setsid "$VLLM_ENV/bin/python" "$VLLM_ENV/bin/vllm" serve "$ckpt" \
    --host 0.0.0.0 --port "$port" --served-model-name "$SERVED_MODEL" \
    --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 --enforce-eager --language-model-only \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > "$logf" 2>&1 &
  echo $!
}
declare -a VPIDS=()
for g in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((8800 + g))
  ckpt="$MODEL_PATH"
  if [ -n "${MPHI_PATH:-}" ] && [ "$g" -eq 0 ]; then ckpt="$MPHI_PATH"; fi
  log "replica $g serves: $ckpt"
  VPIDS+=("$(serve_replica "$g" "$port" "$ckpt" "$OUT_DIR/vllm-$g.log")")
done
trap 'for p in "${VPIDS[@]:-}"; do kill -9 -- "-$p" 2>/dev/null || kill -9 "$p" 2>/dev/null || true; done' EXIT
for g in $(seq 0 $((N_REPLICAS - 1))); do
  port=$((8800 + g)); ok=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "${VPIDS[$g]}" 2>/dev/null || { log "replica $g died"; tail -40 "$OUT_DIR/vllm-$g.log"; exit 1; }
    sleep 5
  done
  [ "$ok" = 1 ] || { log "replica $g not ready"; exit 1; }
  log "replica $g ready (:$port)"
done

cd "$REPO/src"
export OPENAI_API_KEY=EMPTY

# --- propose (instance-wise: K candidates per task) --- #
# With split serving, propose targets ONLY replica 0 (M_phi); otherwise all.
PROPOSE_REPLICAS="$N_REPLICAS"
[ -n "${MPHI_PATH:-}" ] && PROPOSE_REPLICAS=1
# LEAK GUARD: the optional analysis sub-agents must run on the FROZEN M0, never
# on the trained proposer M_phi. Under split serving replica 0 = M_phi, so point
# analysis at a frozen replica (8801). Without split serving every replica is
# the frozen base, so analysis falls back to the propose endpoint (also frozen).
if [ -n "${MPHI_PATH:-}" ] && [ "$N_REPLICAS" -gt 1 ]; then
  export SAH_ANALYSIS_BASE_URL="http://127.0.0.1:8801/v1"   # replica 1 = frozen M0
fi
log "propose: task(s) [$TASKS], K=$K, H1-agent runs across $PROPOSE_REPLICAS replica(s)"
python3 -m outer.rounds.outer_round propose \
  --round-dir "$ROUND_DIR" --round "$ROUND_ID" --k "$K" \
  --tasks $TASKS \
  ${BASES_FILE:+--bases-file "$BASES_FILE"} \
  --n-replicas "$PROPOSE_REPLICAS" --model "$SERVED_MODEL" \
  --max-evals "$MAX_EVALS" ${PROPOSER_SEED:+--seed "$PROPOSER_SEED"} \
  ${SEED_PROGRAMS_FILE:+--seed-programs-file "$SEED_PROGRAMS_FILE"} \
  ${FORCE_TOOL_FRAC:+--force-tool-frac "$FORCE_TOOL_FRAC"} \
  ${FEEDBACK_FILE:+--feedback-file "$FEEDBACK_FILE"} \
  --parallel "${PROPOSE_PAR:-8}" \
  2>&1 | tee "$ROUND_DIR/propose.log"

# The proposer and executor must see the same exact program parent.  propose
# snapshots the mutable cross-round registry before any H1 call; every H2 slot
# below consumes only that round-local snapshot.
ROUND_SEED_PROGRAMS_FILE="$ROUND_DIR/seed_programs_in.json"
[ -s "$ROUND_SEED_PROGRAMS_FILE" ] || {
  echo "missing immutable seed-program snapshot: $ROUND_SEED_PROGRAMS_FILE" >&2
  exit 2
}
python3 - "$ROUND_DIR/round.json" "$ROUND_SEED_PROGRAMS_FILE" <<'PY'
import hashlib, json, sys
meta = json.load(open(sys.argv[1]))
raw = open(sys.argv[2], "rb").read()
expected = (meta.get("cross_round_inputs", {}).get("seed_programs", {})
            .get("snapshot_sha256"))
observed = hashlib.sha256(raw).hexdigest()
if not expected or observed != expected:
    raise SystemExit(
        f"immutable seed-program snapshot mismatch: expected={expected} observed={observed}"
    )
payload = json.loads(raw)
if not isinstance(payload, dict):
    raise SystemExit("immutable seed-program snapshot is not a JSON object")
PY

# --- rollouts: one process per declared H2 slot, bounded pool ------------ #
# Legacy rounds contain only valid candidate slots.  The inference-16 protocol
# uses exactly K=8 H2 slots: an invalid/duplicate H1 proposal spends its paired
# H2 slot on the incoming task-local incumbent harness.  The collector still
# attributes that H1 row as invalid (-1); the fallback is budget accounting,
# not a way to launder an invalid proposal into a positive training example.
fixed_args=()
if [ "${SAH_FIXED_INFERENCE_SLOTS:-0}" = 1 ]; then
  fixed_args+=(--fixed-slots)
  # Under the fail-closed proposal gate every charged slot must hold a valid
  # candidate; the incumbent-fallback pathway is disabled as a second lock.
  [ "${SAH_PROPOSAL_GATE:-off}" != all ] || fixed_args+=(--forbid-fallback)
  [ -z "${SCREEN_EVALS:-}" ] || {
    echo "fixed inference slots are incompatible with cascade SCREEN_EVALS" >&2
    exit 2
  }
fi
if [ "${SAH_PAIRED_CONTROLS:-0}" = 1 ] && [ -n "${SCREEN_EVALS:-}" ]; then
  echo "paired parent controls are not yet compatible with cascade SCREEN_EVALS" >&2
  exit 2
fi
python3 -m outer.reward.trajectory_budget --round-dir "$ROUND_DIR" "${fixed_args[@]}"
mapfile -t PAIRS < <(python3 - "$ROUND_DIR/h2_slot_plan.json" <<'PY'
import json, sys
for row in json.load(open(sys.argv[1]))["slots"]:
    print("\t".join((
        row["task_id"], str(row["k"]), row["h2_harness_dir"],
        "1" if row["h2_slot_mode"] == "incumbent_fallback" else "0",
    )))
PY
)
MAX_PAR="${ROLLOUT_PAR:-8}"
fallback_count=$(python3 - "$ROUND_DIR/h2_slot_plan.json" <<'PY'
import json, sys
print(sum(row["h2_slot_mode"] == "incumbent_fallback"
          for row in json.load(open(sys.argv[1]))["slots"]))
PY
)
log "rollouts: ${#PAIRS[@]} H2 slots ($fallback_count incumbent fallbacks), <= $MAX_PAR concurrent"
mkdir -p "$ROUND_DIR/rollout_logs"
# With split serving, M_phi (replica 0) is no longer needed after propose:
# restart replica 0 with the FROZEN BASE so rollouts get all N replicas.
# Robustness (step-5 lesson): kill the whole process GROUP, wait for GPU 0
# memory to actually free, and if the new server fails to come up FALL BACK
# to replicas 1..N-1 — a dead port must never stay in the rollout rotation.
RB=8800; RN="$N_REPLICAS"
if [ -n "${MPHI_PATH:-}" ]; then
  log "propose done — restarting replica 0 with frozen base for rollouts"
  kill -9 -- "-${VPIDS[0]}" 2>/dev/null || kill -9 "${VPIDS[0]}" 2>/dev/null || true
  freed=0
  for _ in $(seq 1 36); do
    mem=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9)
    [ -n "$mem" ] && [ "$mem" -lt 4096 ] && { freed=1; break; }
    sleep 5
  done
  log "GPU0 freed=$freed (mem=${mem:-?}MiB)"
  VPIDS[0]="$(serve_replica 0 8800 "$MODEL_PATH" "$OUT_DIR/vllm-0-rollout.log")"
  r0=0
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:8800/v1/models" >/dev/null 2>&1 && { r0=1; break; }
    kill -0 "${VPIDS[0]}" 2>/dev/null || break
    sleep 5
  done
  if [ "$r0" = 1 ]; then
    log "replica 0 (frozen base) ready — rollouts use all $N_REPLICAS replicas"
  else
    log "replica 0 restart FAILED — falling back to replicas 1..$((N_REPLICAS-1))"
    kill -9 -- "-${VPIDS[0]}" 2>/dev/null || true
    RB=8801; RN=$((N_REPLICAS - 1))
  fi
fi
run_rollout(){ # tid k evals logsuffix harness_dir fallback_flag output_group repeat
  local tid="$1" k="$2" evals="$3" sfx="$4" cdir="$5" fallback="$6"
  local output_group="${7:-rollouts}"
  local repeat="${8:-0}"
  local port=$((RB + ROLL_IDX % RN)); ROLL_IDX=$((ROLL_IDX + 1))
  local logical_index="${LOGICAL_ROUND_INDEX:-0}"
  # Reserve a non-overlapping seed block per logical comparison batch. Each
  # repeat layer shifts by K; candidate and parent-control use the same layer.
  local decode_seed=$(( ${H2_SEED_BASE:-200000} + logical_index * PAIR_SEED_STRIDE + repeat * K + k ))
  [ "$fallback" = 1 ] && log "  H2 slot $tid/cand$(printf '%02d' "$k"): incumbent fallback"
  OPENAI_BASE_URL="http://127.0.0.1:$port/v1" \
    timeout --foreground --kill-after=60s "${ROLLOUT_WALL_TIMEOUT:-10800}s" \
    python3 -m inner.cli.run_baseline \
    --harness-dir "$cdir" --ids "$tid" \
    --base-url "http://127.0.0.1:$port/v1" --model "$SERVED_MODEL" \
    --max-evals "$evals" ${EVAL_TIMEOUT:+--eval-timeout "$EVAL_TIMEOUT"} \
    --seed "$decode_seed" \
    --seed-programs-file "$ROUND_SEED_PROGRAMS_FILE" \
    --eval-python python3 --require-trajectory \
    --out "$ROUND_DIR/$output_group/$tid/cand$(printf '%02d' "$k")" \
    > "$ROUND_DIR/rollout_logs/${tid}-cand$(printf '%02d' "$k")${sfx}.log" 2>&1 &
}
log "trajectory watchdog=${ROLLOUT_WALL_TIMEOUT:-10800}s (launched work remains charged)"
ROLL_IDX=0; rc=0
if [ -n "${SCREEN_EVALS:-}" ]; then
  # Successive halving: every candidate gets a cheap SCREEN_EVALS rollout,
  # only the top PROMOTE go on to a full MAX_EVALS rollout. load_rollout_score
  # takes the max over run subdirs, so collect automatically sees the best.
  log "cascade pass 1: ${#PAIRS[@]} candidates @ $SCREEN_EVALS evals"
  for pair in "${PAIRS[@]:-}"; do
    [ -n "$pair" ] || continue
    IFS=$'\t' read -r tid k cdir fallback <<< "$pair"
    while (( $(jobs -rp | wc -l) >= MAX_PAR )); do wait -n || rc=1; done
    run_rollout "$tid" "$k" "$SCREEN_EVALS" "-screen" "$cdir" "$fallback"
  done
  while (( $(jobs -rp | wc -l) > 0 )); do wait -n || rc=1; done
  mapfile -t PROMOTED < <(python3 "$REPO/scripts/runtime/cascade_promote.py" "$ROUND_DIR" "${PROMOTE:-4}")
  log "cascade pass 2: ${#PROMOTED[@]} promoted @ $MAX_EVALS evals"
  for pair in "${PROMOTED[@]:-}"; do
    [ -n "$pair" ] || continue
    while (( $(jobs -rp | wc -l) >= MAX_PAR )); do wait -n || rc=1; done
    tid="${pair%%:*}"; k="${pair##*:}"
    cdir=$(printf '%s/tasks/%s/cand%02d' "$ROUND_DIR" "$tid" "$k")
    run_rollout "$tid" "$k" "$MAX_EVALS" "-full" "$cdir" 0
  done
  while (( $(jobs -rp | wc -l) > 0 )); do wait -n || rc=1; done
else
  for pair in "${PAIRS[@]:-}"; do
    [ -n "$pair" ] || continue
    IFS=$'\t' read -r tid k cdir fallback <<< "$pair"
    while (( $(jobs -rp | wc -l) >= MAX_PAR )); do wait -n || rc=1; done
    run_rollout "$tid" "$k" "$MAX_EVALS" "" "$cdir" "$fallback"
  done
  while (( $(jobs -rp | wc -l) > 0 )); do wait -n || rc=1; done
fi
log "rollouts finished (rc=$rc)"
[ "$rc" -eq 0 ] || { log "one or more rollouts failed; refusing to collect"; exit "$rc"; }

if [ "${SAH_PAIRED_CONTROLS:-0}" = 1 ] && [ "$PAIRED_REPEATS" -gt 1 ]; then
  for repeat in $(seq 1 $((PAIRED_REPEATS - 1))); do
    log "paired candidate repeats: layer $repeat/$((PAIRED_REPEATS - 1))"
    rc=0
    for pair in "${PAIRS[@]:-}"; do
      [ -n "$pair" ] || continue
      IFS=$'\t' read -r tid k cdir fallback <<< "$pair"
      while (( $(jobs -rp | wc -l) >= MAX_PAR )); do wait -n || rc=1; done
      run_rollout "$tid" "$k" "$MAX_EVALS" "-repeat${repeat}" \
        "$cdir" "$fallback" "rollouts" "$repeat"
    done
    while (( $(jobs -rp | wc -l) > 0 )); do wait -n || rc=1; done
    [ "$rc" -eq 0 ] || {
      log "one or more repeated candidate rollouts failed"
      exit "$rc"
    }
  done
fi
python3 "$REPO/scripts/runtime/audit_trajectories.py" "$ROUND_DIR/rollouts"

# Causal H1 credit: rerun the incoming parent H2 under the exact decode seed,
# task/program snapshot, model, budget, and evaluator settings used by each
# candidate slot.  This separates a harness-induced behavior change from an
# executor discovery already available without that candidate harness.
if [ "${SAH_PAIRED_CONTROLS:-0}" = 1 ]; then
  mapfile -t CONTROL_PAIRS < <(python3 - "$ROUND_DIR/round.json" "$ROUND_DIR/h2_slot_plan.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
plan = json.load(open(sys.argv[2]))
for row in plan["slots"]:
    task_id = row["task_id"]
    print("\t".join((
        task_id,
        str(row["k"]),
        str(meta["per_task"][task_id]["base_package"]),
    )))
PY
  )
  log "paired controls: ${#CONTROL_PAIRS[@]} parent H2s x $PAIRED_REPEATS matched seed(s) @ $MAX_EVALS evals"
  rc=0
  for repeat in $(seq 0 $((PAIRED_REPEATS - 1))); do
    for pair in "${CONTROL_PAIRS[@]:-}"; do
      [ -n "$pair" ] || continue
      IFS=$'\t' read -r tid k parent_dir <<< "$pair"
      while (( $(jobs -rp | wc -l) >= MAX_PAR )); do wait -n || rc=1; done
      run_rollout "$tid" "$k" "$MAX_EVALS" \
        "-parent-control-repeat${repeat}" "$parent_dir" 0 \
        "paired_controls" "$repeat"
    done
    while (( $(jobs -rp | wc -l) > 0 )); do wait -n || rc=1; done
  done
  log "paired controls finished (rc=$rc)"
  [ "$rc" -eq 0 ] || {
    log "one or more parent controls failed; refusing unpaired credit"
    exit "$rc"
  }
  python3 "$REPO/scripts/runtime/audit_trajectories.py" "$ROUND_DIR/paired_controls"
fi

# --- collect --- #
python3 -m outer.rounds.outer_round collect --round-dir "$ROUND_DIR" 2>&1 | tee "$ROUND_DIR/collect.log"
log "round $ROUND_ID done -> $ROUND_DIR"
