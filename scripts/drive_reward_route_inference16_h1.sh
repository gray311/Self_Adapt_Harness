#!/usr/bin/env bash
# Run one proposer-weight or context-only task under the inference-16 protocol.
# This file is intentionally guarded: configuration review alone cannot submit
# a job.  After author confirmation set RR_RUN_CONFIRMED=YES explicitly.
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
source "$SAH/scripts/reward_route_inference16_config.sh"
rr_assert_config

[ "${RR_RUN_CONFIRMED:-NO}" = YES ] || {
  echo "refusing to run: wait for author confirmation, then set RR_RUN_CONFIRMED=YES" >&2
  exit 3
}

METHOD="${1:?usage: drive_reward_route_inference16_h1.sh proposer|context TASK_TAG}"
TASK_TAG="${2:?task tag: erdos|ac2|hadamard|eplb}"
case " ${RR_TASK_TAGS[*]} " in
  *" $TASK_TAG "*) ;;
  *) echo "task $TASK_TAG is outside the frozen inference16 figure set" >&2; exit 2 ;;
esac
case "$METHOD" in proposer|context) ;; *) echo "invalid method: $METHOD" >&2; exit 2 ;; esac
TASK=$(rr_task_id "$TASK_TAG")
ROUND_BASE=$(rr_round_base "$METHOD" "$TASK_TAG")

ROOT="$RUN_ROOT/self_adapt_harness/$RR_PROTOCOL_ID"
WS="$ROOT/${METHOD}_${TASK_TAG}"
OUT_TAG="${RR_PROTOCOL_ID}-${METHOD}-${TASK_TAG}"
OUT="$RUN_ROOT/self_adapt_harness/outer-$OUT_TAG"
mkdir -p "$ROOT"
exec 9>"$ROOT/.${METHOD}_${TASK_TAG}.controller.lock"
if ! flock -n 9; then
  echo "another inference16 controller owns $METHOD/$TASK_TAG" >&2
  exit 0
fi
[ ! -e "$WS/run_manifest.json" ] && [ ! -e "$OUT/round$(printf '%03d' "$ROUND_BASE")" ] || {
  echo "immutable inference16 namespace already exists for $METHOD/$TASK_TAG" >&2
  exit 2
}
mkdir -p "$WS" "$OUT"
python3 "$SAH/scripts/runtime/capture_shared_anchor.py" \
  --index "$SAH/results/baseline_h2_20ev_program_index.json" \
  --task "$TASK" --out-dir "$WS/shared_anchor" >/dev/null

worker_sha=$(sha256sum "$SAH/src/inner/evaluation/_eval_worker.py" | awk '{print $1}')
initial_h2_sha=$(python3 "$SAH/scripts/runtime/hash_h2_package.py" "$SAH/src/inner/harness")
guard_file="$RUN_ROOT/self_adapt_harness/protocol_guards.ok"
grep -q "^eval_worker_sha256=${worker_sha}$" "$guard_file"
if [ "$TASK_TAG" = eplb ]; then
  grep -q '^eplb_topology_guard=ok$' "$guard_file"
fi

python3 - "$WS/round000_bases.json" "$WS/run_manifest.json" "$SAH" \
  "$TASK" "$METHOD" "$TASK_TAG" "$ROUND_BASE" "$RR_PROTOCOL_ID" \
  "$worker_sha" "$guard_file" "$initial_h2_sha" "$WS/shared_anchor/manifest.json" <<'PY'
import hashlib, json, os, sys, time
(base_path, manifest_path, repo, task, method, tag, round_base, protocol,
 worker_sha, guard_file, initial_h2_sha, anchor_manifest) = sys.argv[1:]
baseline = json.load(open(f"{repo}/results/baseline_h2_20ev.json"))["baseline"][task]
json.dump({task: {
    "package": f"{repo}/src/inner/harness",
    "score": float(baseline["h2_best"]),
    "seed_score": float(baseline["seed"]),
}}, open(base_path, "w"), indent=2)
sources = [
    f"{repo}/scripts/reward_route_inference16_config.sh",
    f"{repo}/scripts/drive_reward_route_inference16_h1.sh",
    f"{repo}/scripts/fresh_campaign.sh",
    f"{repo}/scripts/context_ablation.sh",
    f"{repo}/scripts/outer_round.sbatch",
    f"{repo}/scripts/_outer_round_worker.sh",
    f"{repo}/src/outer/rounds/outer_round.py",
    f"{repo}/src/outer/reward/trajectory_budget.py",
    f"{repo}/src/outer/reward/rewards.py",
    f"{repo}/src/outer/harness/cordis.yml",
    f"{repo}/cordis/plugins/sah-bridge.mjs",
    f"{repo}/src/cordis_runtime/runner.py",
    f"{repo}/results/baseline_h2_20ev.json",
    f"{repo}/results/human_best_references.json",
]
json.dump({
    "schema": 1,
    "status": "running",
    "protocol": protocol,
    "method": method,
    "task": task,
    "task_tag": tag,
    "round_base": int(round_base),
    "rounds": 19,
    "trajectory_axis": "generated_agent_trajectory",
    "h1_trajectories_per_round": 8,
    "h2_trajectories_per_round": 8,
    "total_trajectories_per_round": 16,
    "shared_anchor_x": 1,
    "planned_final_x": 305,
    "curated_notes_allowed": False,
    "historical_program_imported": False,
    "controller_job": os.environ.get("SLURM_JOB_ID"),
    "eval_worker_sha256": worker_sha,
    "initial_h2_sha256": initial_h2_sha,
    "shared_anchor": json.load(open(anchor_manifest))["tasks"][task],
    "h1_max_iterations": 24,
    "program_ratchet_mode": "strict_single",
    "minimum_trainable_h1_rows": 4,
    "post_submit_reviewer_model_calls": 0,
    "protocol_guard_file": guard_file,
    "source_sha256": {
        path: hashlib.sha256(open(path, "rb").read()).hexdigest()
        for path in sources
    },
    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}, open(manifest_path, "w"), indent=2)
PY

export RUNTIME_SOURCE_MANIFEST="$WS/runtime_source_manifest.json"
if [ -s "$RUNTIME_SOURCE_MANIFEST" ]; then
  python3 "$SAH/scripts/runtime/provenance.py" verify \
    --manifest "$RUNTIME_SOURCE_MANIFEST"
else
  python3 "$SAH/scripts/runtime/provenance.py" snapshot --repo "$SAH" \
    --manifest "$RUNTIME_SOURCE_MANIFEST" --snapshot-dir "$WS/source_snapshot"
fi

export OUT_TAG K="$RR_H1_TRAJECTORIES" MAX_EVALS="$RR_MAX_EVALS"
export EVAL_TIMEOUT="$RR_EVAL_TIMEOUT" ROLLOUT_WALL_TIMEOUT="$RR_ROLLOUT_WALL_TIMEOUT"
export FTF="$RR_FORCE_TOOL_FRAC" LOGICAL_SEED_BASE="$RR_LOGICAL_SEED_BASE"
export H2_SEED_BASE="$RR_H2_SEED_BASE"
export SAH_PROGRAM_RATCHET_MODE="$RR_PROGRAM_RATCHET_MODE"
export SAH_REQUIRE_STRICT_RATCHET=1
export SAH_TASK_TEXT_ENFORCE=1
export MIN_TRAINABLE_ROWS="$RR_MIN_TRAINABLE_H1_ROWS"
export SAH_FIXED_INFERENCE_SLOTS=1 SAH_ANALYSIS_REQUIRED=0
export SAH_ADV=v3 SAH_HIST_LAMBDA=0.3 SAH_ALPHA=0.3
export SAH_LEAK_NEUTRALIZE=1 SAH_CHAMPION_ISOLATION=1
export NO_CURATED_NOTES=1 ARCHIVE_MIX=0 PLATEAU_ROUNDS=1 SKIP_FINAL_TRAIN=1
export EVAL_REPEATS=1 SAH_MIN_ITERS=0 SAH_SEQUENTIAL=0 SAH_SEQ_MAX_SHARED=6
export PROPOSE_PAR=8 ROLLOUT_PAR=8 SAH_ANALYSIS_PERF=1 SAH_ANALYSIS_DESIGN=1
export NO_TRAIN=0
export POST_BATCH_HOOK="$SAH/scripts/refresh_reward_route_inference16.sh"
export LORA_RANK="$RR_LORA_RANK" LORA_ALPHA="$RR_LORA_ALPHA"
export NUM_EPOCH="$RR_EPOCHS" LR="$RR_LR" KL_COEF="$RR_KL"
unset SCREEN_EVALS PROMOTE NO_INHERIT RESUME_BASES RESUME_PHI RESUME_CKPT

if [ "$TASK_TAG" = eplb ]; then
  grep -q '^eplb_topology_guard=ok$' "$RUN_ROOT/self_adapt_harness/protocol_guards.ok"
fi

set +e
if [ "$METHOD" = proposer ]; then
  export SAH_ANALYSIS=0
  export STAG_PREFIX="rri16_${TASK_TAG}_proposer"
  bash "$SAH/scripts/fresh_campaign.sh" \
    "$TASK" "$RR_ROUNDS" "$ROUND_BASE" "$RR_FORCE_TOOL_FRAC" "$WS"
  driver_rc=$?
else
  unset MPHI
  export CTX_TASKS="$TASK" TRAIN_PHI=0 USE_ANALYST=1 SAH_ANALYSIS_REQUIRED=1
  bash "$SAH/scripts/context_ablation.sh" "$RR_ROUNDS" "$ROUND_BASE" "$WS"
  driver_rc=$?
fi
set -e

completed=$(find "$OUT" -maxdepth 2 -type f -name round_summary.json | wc -l)
status=incomplete
if [ "$driver_rc" -eq 0 ] && [ "$completed" -eq "$RR_ROUNDS" ]; then
  status=complete
fi
python3 - "$WS/run_manifest.json" "$status" "$driver_rc" "$completed" <<'PY'
import json, os, sys, time
path, status, driver_rc, completed = sys.argv[1:]
payload = json.load(open(path))
payload.update({
    "status": status,
    "driver_exit_code": int(driver_rc),
    "completed_rounds": int(completed),
    "last_driver_exit_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
})
if status == "complete":
    payload["completed_at"] = payload["last_driver_exit_at"]
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(payload, handle, indent=2)
os.replace(tmp, path)
PY
if [ "$status" = complete ]; then
  touch "$WS/COMPLETE"
  exit 0
fi
echo "inference16 $METHOD/$TASK_TAG incomplete: rc=$driver_rc rounds=$completed/$RR_ROUNDS" >&2
exit 1
