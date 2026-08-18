#!/usr/bin/env bash
# One-shot submission of the 4 tasks x 3 routes under inference16-v1.
# The nine CPU controllers submit sequential GPU stages and remain alive so
# Slurm—not a login shell—owns the long-running experiment state machine.
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
ROOT="$RUN_ROOT/self_adapt_harness/reward-route-inference16-v1"
RECEIPT="$SAH/results/reward_route_inference16_submission.json"
GUARD="$RUN_ROOT/self_adapt_harness/protocol_guards.ok"
: "${RR_RUN_CONFIRMED:?set RR_RUN_CONFIRMED=YES after author approval}"
[ "$RR_RUN_CONFIRMED" = YES ] || { echo "RR_RUN_CONFIRMED must equal YES" >&2; exit 3; }
[ ! -e "$RECEIPT" ] || { echo "submission receipt already exists: $RECEIPT" >&2; exit 2; }
[ ! -e "$ROOT/executor/run_manifest.json" ] || { echo "executor namespace already initialized" >&2; exit 2; }

cd "$SAH"
PYTHONPATH=.:src python3 scripts/analysis/audits/reward_route_inference16.py >/dev/null
PYTHONPATH=.:src python3 -m unittest discover -s tests -p 'test_inference16_protocol.py' >/dev/null
PYTHONPATH=.:src python3 -m unittest discover -s tests -p 'test_outer_rewards.py' >/dev/null
worker_sha=$(sha256sum src/inner/evaluation/_eval_worker.py | awk '{print $1}')
grep -q "^eval_worker_sha256=${worker_sha}$" "$GUARD"
grep -q '^eplb_topology_guard=ok$' "$GUARD"
grep -q '^prism_success_guard=ok$' "$GUARD"
grep -q '^txn_legality_guard=ok$' "$GUARD"

for method in proposer context; do
  for tag in erdos ac2 hadamard eplb; do
    [ ! -e "$ROOT/${method}_${tag}/run_manifest.json" ] || {
      echo "namespace already initialized: $method/$tag" >&2; exit 2; }
  done
done

# The user previously budgeted at most 56 concurrently working GPUs.  Four
# frozen tasks create twelve independent 4-GPU pipelines (48 GPUs), but older
# campaign controllers are still alive and can submit more GPU stages.  Make
# every new controller wait for the currently queued/running CPU controllers;
# dependency-pending jobs consume no GPU and avoid overlapping the generations.
mapfile -t predecessor_jobs < <(
  squeue -h -u "$USER" -p cpu -o '%i|%j' | \
    awk -F'|' '$2 !~ /^rri16-/ {print $1}'
)
dependency_args=()
dependency_spec=""
if [ "${#predecessor_jobs[@]}" -gt 0 ]; then
  dependency_spec="afterany:$(IFS=:; echo "${predecessor_jobs[*]}")"
  dependency_args=(--dependency="$dependency_spec")
fi

submit_retry(){
  local label="$1"; shift
  local attempt out
  for attempt in $(seq 1 20); do
    if out=$(timeout 60s sbatch --parsable "$@" 2>/dev/null) && [[ "$out" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "$out"
      return 0
    fi
    echo "$label submission attempt $attempt failed; retrying in 30s" >&2
    sleep 30
  done
  echo "$label submission failed after 20 attempts" >&2
  return 1
}

tmp="$RECEIPT.tmp.$$"
python3 - "$tmp" "$worker_sha" "$dependency_spec" "${predecessor_jobs[*]:-}" <<'PY'
import json, os, sys, time
path, worker_sha, dependency, predecessors = sys.argv[1:]
json.dump({
    "schema": 1,
    "protocol": "reward-route-inference16-v1",
    "status": "submitting",
    "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "submitted_by": os.environ.get("USER"),
    "eval_worker_sha256": worker_sha,
    "gpu_concurrency_policy": (
        "wait for pre-existing CPU controllers; then at most twelve 4-GPU "
        "task-route pipelines (48 GPUs)"
    ),
    "controller_dependency": dependency or None,
    "predecessor_controller_jobs": [int(x) for x in predecessors.split() if x],
    "controllers": [],
}, open(path, "w"), indent=2)
PY
mv "$tmp" "$RECEIPT"

append_job(){
  python3 - "$RECEIPT" "$1" "$2" "$3" <<'PY'
import json, os, sys
path, job, method, task = sys.argv[1:]
payload = json.load(open(path))
payload["controllers"].append({"job_id": int(job), "method": method, "task": task})
tmp = path + ".tmp"
json.dump(payload, open(tmp, "w"), indent=2)
os.replace(tmp, path)
PY
}

# Interleave routes so scheduler priority does not systematically favor one H1
# method.  Executor has one controller which launches all five task pipelines.
executor_job=$(submit_retry executor \
  --job-name=rri16-exec \
  "${dependency_args[@]}" \
  --export=ALL,RR_RUN_CONFIRMED=YES \
  scripts/drive_reward_route_inference16_executor.sbatch)
append_job "$executor_job" executor all

for tag in erdos ac2 hadamard eplb; do
  short="$tag"; [ "$tag" = hadamard ] && short=had
  prop_job=$(submit_retry "proposer/$tag" \
    --job-name="rri16-p-${short}" \
    "${dependency_args[@]}" \
    --export="ALL,RR_RUN_CONFIRMED=YES,METHOD=proposer,TASK_TAG=$tag" \
    scripts/drive_reward_route_inference16_h1.sbatch)
  append_job "$prop_job" proposer "$tag"
  ctx_job=$(submit_retry "context/$tag" \
    --job-name="rri16-c-${short}" \
    "${dependency_args[@]}" \
    --export="ALL,RR_RUN_CONFIRMED=YES,METHOD=context,TASK_TAG=$tag" \
    scripts/drive_reward_route_inference16_h1.sbatch)
  append_job "$ctx_job" context "$tag"
done

python3 - "$RECEIPT" <<'PY'
import json, os, sys, time
path = sys.argv[1]
payload = json.load(open(path))
payload["status"] = "submitted"
payload["controller_count"] = len(payload["controllers"])
payload["submission_finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
tmp = path + ".tmp"
json.dump(payload, open(tmp, "w"), indent=2)
os.replace(tmp, path)
print(json.dumps(payload, indent=2))
PY
