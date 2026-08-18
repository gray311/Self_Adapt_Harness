#!/usr/bin/env bash
# Run the four fixed-16-H2 executor-update pipelines after explicit approval.
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

export STATE_ROOT="$RUN_ROOT/self_adapt_harness/$RR_PROTOCOL_ID/executor"
export H2_SEED_BASE="$RR_H2_SEED_BASE"
mkdir -p "$STATE_ROOT"
python3 "$SAH/scripts/runtime/capture_shared_anchor.py" \
  --index "$SAH/results/baseline_h2_20ev_program_index.json" \
  --task eft__math__erdos_min_overlap \
  --task eft__math__second_autocorr_ineq \
  --task eft__math__hadamard_maximal_det \
  --task adrs__eplb --out-dir "$STATE_ROOT/shared_anchor" >/dev/null
worker_sha=$(sha256sum "$SAH/src/inner/evaluation/_eval_worker.py" | awk '{print $1}')
export FIXED_H2_SHA256
FIXED_H2_SHA256=$(python3 "$SAH/scripts/runtime/hash_h2_package.py" "$SAH/src/inner/harness")
guard_file="$RUN_ROOT/self_adapt_harness/protocol_guards.ok"
grep -q "^eval_worker_sha256=${worker_sha}$" "$guard_file"
grep -q '^eplb_topology_guard=ok$' "$guard_file"

python3 - "$STATE_ROOT/run_manifest.json" "$SAH" "$RR_PROTOCOL_ID" \
  "$worker_sha" "$guard_file" "$FIXED_H2_SHA256" \
  "$STATE_ROOT/shared_anchor/manifest.json" <<'PY'
import hashlib, json, os, sys, time
path, repo, protocol, worker_sha, guard_file, fixed_h2_sha, anchor_manifest = sys.argv[1:]
sources = [
    f"{repo}/scripts/reward_route_inference16_config.sh",
    f"{repo}/scripts/drive_reward_route_inference16_executor.sh",
    f"{repo}/scripts/drive_ttt_executor_12h.sh",
    f"{repo}/scripts/ttt_executor_eval.sbatch",
    f"{repo}/scripts/runtime/ttt_discover_prepare.py",
    f"{repo}/scripts/submit_ttt_executor_update.sh",
    f"{repo}/src/inner/evaluation/_eval_worker.py",
    f"{repo}/results/baseline_h2_20ev.json",
    f"{repo}/results/human_best_references.json",
]
try:
    payload = json.load(open(path))
except FileNotFoundError:
    payload = {
        "schema": 1,
        "protocol": protocol,
        "method": "update executor weights (TTT-style reference)",
        "tasks": ["erdos", "ac2", "hadamard", "eplb"],
        "rounds": 19,
        "evaluated_weight_updates": 18,
        "trajectory_axis": "generated_agent_trajectory",
        "h1_trajectories_per_round": 0,
        "h2_trajectories_per_round": 16,
        "total_trajectories_per_round": 16,
        "shared_anchor_x": 1,
        "planned_final_x": 305,
        "no_top_up": True,
        "historical_program_imported": False,
        "fixed_h2_sha256": fixed_h2_sha,
        "shared_anchors": json.load(open(anchor_manifest))["tasks"],
        "source_sha256": {
            p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in sources
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
elif payload.get("protocol") != protocol:
    raise SystemExit(f"refusing incompatible executor namespace: {path}")
payload.update({
    "status": "running",
    "controller_job": os.environ.get("SLURM_JOB_ID"),
    "eval_worker_sha256": worker_sha,
    "protocol_guard_file": guard_file,
    "fixed_h2_sha256": fixed_h2_sha,
    "shared_anchors": json.load(open(anchor_manifest))["tasks"],
    "last_started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
})
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(payload, handle, indent=2)
os.replace(tmp, path)
PY

export RUNTIME_SOURCE_MANIFEST="$STATE_ROOT/runtime_source_manifest.json"
if [ -s "$RUNTIME_SOURCE_MANIFEST" ]; then
  python3 "$SAH/scripts/runtime/provenance.py" verify \
    --manifest "$RUNTIME_SOURCE_MANIFEST"
else
  python3 "$SAH/scripts/runtime/provenance.py" snapshot --repo "$SAH" \
    --manifest "$RUNTIME_SOURCE_MANIFEST" --snapshot-dir "$STATE_ROOT/source_snapshot"
fi

export TASK_SET=curve4 K="$RR_H2_TRAJECTORIES_EXECUTOR_ROUTE"
export UPDATES="$RR_EXECUTOR_POST_UPDATE_EVALS" MODEL_TAG=rri16e
export FIXED_LAUNCHED_BATCH=1 MIN_TRAIN_USABLE=8
export MAX_SUPPLEMENT_LAUNCHES=0 AHC_SUPPLEMENT_K=0
export PARTIAL_MIN_USABLE=8 PARTIAL_AFTER_SECONDS=999999
export TTT_MATCH_PROPOSER_TRAIN=1 TTT_LR="$RR_LR" TTT_KL_COEF="$RR_KL"
export PLOT_LAYOUT=none
export DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(( $(date +%s) + 1209600 ))}"
export DRIVER_LOCK_SUFFIX=_inference16
export POST_BATCH_HOOK="$SAH/scripts/refresh_reward_route_inference16.sh"
unset TASK_FILTER MAX_PARALLEL AHC_CASE_WORKERS TTT_PREFIX

set +e
bash "$SAH/scripts/drive_ttt_executor_12h.sh"
driver_rc=$?
set -e

python3 - "$STATE_ROOT/run_manifest.json" "$STATE_ROOT" "$driver_rc" <<'PY'
import json, os, sys, time
path, root, driver_rc = sys.argv[1:]
tags = ("erdos", "ac2", "hadamard", "eplb")
counts = {}
for tag in tags:
    curve = os.path.join(root, tag, "curve.jsonl")
    try:
        counts[tag] = sum(bool(line.strip()) for line in open(curve))
    except FileNotFoundError:
        counts[tag] = 0
complete = int(driver_rc) == 0 and all(count == 19 for count in counts.values())
payload = json.load(open(path))
payload.update({
    "status": "complete" if complete else "incomplete",
    "driver_exit_code": int(driver_rc),
    "completed_batches": counts,
    "last_driver_exit_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
})
if complete:
    payload["completed_at"] = payload["last_driver_exit_at"]
tmp = path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(payload, handle, indent=2)
os.replace(tmp, path)
raise SystemExit(0 if complete else 1)
PY
touch "$STATE_ROOT/COMPLETE"
