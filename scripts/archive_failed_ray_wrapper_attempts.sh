#!/usr/bin/env bash
# Preserve the two pre-optimizer Ray-wrapper failures after the environment
# relocation repair has passed, then reopen only their canonical slots.

set -euo pipefail
umask 027

if [ "$#" -ne 4 ]; then
  echo "usage: $0 EVOLVE_RUN_DIR MAIN_JOB_ID TTT_JOB_ID REPAIR_JOB_ID" >&2
  exit 2
fi

readonly run_dir="$(readlink -f "$1")"
readonly main_job_id="$2"
readonly ttt_job_id="$3"
readonly repair_job_id="$4"
readonly ttt_dir="$run_dir/baselines/update_executor_ttt"
readonly main_attempt="$run_dir/training_attempts/ray_wrapper_relocation_${main_job_id}"
readonly ttt_attempt="$ttt_dir/training_attempts/ray_wrapper_relocation_${ttt_job_id}"
readonly repair_dir="$run_dir/../../preflight/train-env-relocation-repair-${repair_job_id}"

case "$run_dir" in
  /lustre/fsw/portfolios/av/users/yingzim/runs/campaigns/* | \
  /lustre/fsw/portfolios/av/projects/av_alpamayo_reasoning/users/yingzim/runs/campaigns/*) ;;
  *) echo "refusing unexpected run directory: $run_dir" >&2; exit 2 ;;
esac
[[ "$main_job_id" =~ ^[0-9]+$ ]]
[[ "$ttt_job_id" =~ ^[0-9]+$ ]]
[[ "$repair_job_id" =~ ^[0-9]+$ ]]

readonly main_phase="$run_dir/phases/training_smoke"
readonly ttt_phase="$ttt_dir/phases/training"
main_failure_evidence="$main_phase"
ttt_failure_evidence="$ttt_phase"
if [ ! -e "$main_failure_evidence" ]; then
  main_failure_evidence="$main_attempt/phases/training_smoke"
fi
if [ ! -e "$ttt_failure_evidence" ]; then
  ttt_failure_evidence="$ttt_attempt/phases/training"
fi
readonly main_failure_evidence ttt_failure_evidence

test -s "$run_dir/EVOLVE_PASSED"
test -s "$ttt_dir/TTT_PREPARE_PASSED"
test -s "$repair_dir/PASSED"
test -s "$main_failure_evidence/FAILED"
test -s "$ttt_failure_evidence/FAILED"
grep -Fxq 'returncode=1' "$main_failure_evidence/FAILED"
grep -Fxq 'returncode=1' "$ttt_failure_evidence/FAILED"
grep -Fq '.partial-6184473/bin/python: not found' "$main_failure_evidence/ray_head.log"

ttt_ray_log="$ttt_dir/training/stage1/ray_head.log"
if [ ! -e "$ttt_ray_log" ]; then
  ttt_ray_log="$ttt_attempt/training/stage1/ray_head.log"
fi
grep -Fq '.partial-6184473/bin/python: not found' "$ttt_ray_log"

test ! -e "$run_dir/TRAINING_SMOKE_PASSED"
test ! -e "$run_dir/DEBUG_CLOSED_LOOP_PASSED"
test ! -e "$ttt_dir/TTT_BASELINE_SMOKE_PASSED"
test ! -e "$main_attempt/ARCHIVED"
test ! -e "$ttt_attempt/ARCHIVED"

# Empty checkpoint directories are expected because they were created before
# Ray startup. Any checkpoint file means the failure happened later and must not
# be handled by this recovery.
if [ -d "$run_dir/checkpoints/training_smoke" ] \
    && find "$run_dir/checkpoints/training_smoke" -type f -print -quit | grep -q .; then
  echo "main attempt unexpectedly contains checkpoint files" >&2
  exit 3
fi
if [ -d "$ttt_dir/checkpoints" ] \
    && find "$ttt_dir/checkpoints" -type f -print -quit | grep -q .; then
  echo "TTT attempt unexpectedly contains checkpoint files" >&2
  exit 3
fi
test ! -e "$main_failure_evidence/train.log"
test ! -e "$ttt_dir/training/stage1/train.log"
test ! -e "$ttt_attempt/training/stage1/train.log"

mkdir -p "$main_attempt/launchers" "$ttt_attempt"

archive_main() {
  local relative="$1"
  local source="$run_dir/$relative" destination="$main_attempt/$relative"
  if { test -e "$source" || test -L "$source"; } \
      && { test -e "$destination" || test -L "$destination"; }; then
    echo "both canonical and archived main paths exist: $relative" >&2
    exit 4
  fi
  if test -e "$destination" || test -L "$destination"; then
    echo "already archived main path: $relative"
    return
  fi
  test -e "$source" || test -L "$source"
  mkdir -p "$main_attempt/$(dirname "$relative")"
  if [ "$relative" = training_source_snapshot ]; then
    chmod u+w "$source"
  fi
  mv -- "$source" "$destination"
  if [ "$relative" = training_source_snapshot ]; then
    chmod u-w "$destination"
  fi
}

archive_ttt() {
  local relative="$1"
  local source="$ttt_dir/$relative" destination="$ttt_attempt/$relative"
  if { test -e "$source" || test -L "$source"; } \
      && { test -e "$destination" || test -L "$destination"; }; then
    echo "both canonical and archived TTT paths exist: $relative" >&2
    exit 4
  fi
  if test -e "$destination" || test -L "$destination"; then
    echo "already archived TTT path: $relative"
    return
  fi
  test -e "$source" || test -L "$source"
  mkdir -p "$ttt_attempt/$(dirname "$relative")"
  if [ "$relative" = training_source_snapshot ]; then
    chmod u+w "$source"
  fi
  if [ "$relative" = launchers/iad_ttt_baseline_train_inner.sh ]; then
    chmod u+w "$(dirname "$source")"
  fi
  mv -- "$source" "$destination"
  if [ "$relative" = training_source_snapshot ]; then
    chmod u-w "$destination"
  fi
  if [ "$relative" = launchers/iad_ttt_baseline_train_inner.sh ]; then
    chmod u-w "$(dirname "$source")"
  fi
}

for relative in \
  phases/training_smoke \
  phases/merged_serve_smoke \
  training_source_snapshot \
  audits/training_source_snapshot_audit.json \
  audits/canonical_replay_audit.json \
  audits/bounded_replay_audit.json \
  train_runtime_manifest.json \
  train_environment_build_manifest.json \
  checkpoints/training_smoke \
  replay/grpo_replay_train_bounded.jsonl; do
  archive_main "$relative"
done

for launcher in \
  iad_evolve_train_smoke_inner.sh \
  iad_evolve_train_smoke.sbatch \
  iad_training_smoke_audit.py \
  iad_vllm_protocol_probe.py \
  iad_vllm_server_common.sh; do
  archive_main "launchers/$launcher"
done

for relative in \
  phases/training \
  training_source_snapshot \
  audits/training_source_snapshot_audit.json \
  audits/replay_audit.json \
  train_runtime_manifest.json \
  train_environment_build_manifest.json \
  launchers/iad_ttt_baseline_train_inner.sh \
  training \
  checkpoints \
  exports \
  after_rollouts; do
  archive_ttt "$relative"
done

cp "$0" "$main_attempt/archive_script.sh"
cp "$0" "$ttt_attempt/archive_script.sh"
chmod a-w "$main_attempt/archive_script.sh" "$ttt_attempt/archive_script.sh"
printf 'archived_utc=%s\nfailed_job_id=%s\ncause=published_venv_ray_wrapper_referenced_partial_build\nrepair_job_id=%s\nslurm_log=%s\n' \
  "$(date -Is -u)" "$main_job_id" "$repair_job_id" \
  "$run_dir/../../slurm/sah-evolve-train-smoke-${main_job_id}.err" > "$main_attempt/ARCHIVED"
printf 'archived_utc=%s\nfailed_job_id=%s\ncause=published_venv_ray_wrapper_referenced_partial_build\nrepair_job_id=%s\nslurm_log=%s\n' \
  "$(date -Is -u)" "$ttt_job_id" "$repair_job_id" \
  "$run_dir/../../slurm/sah-ttt-train-smoke-${ttt_job_id}.err" > "$ttt_attempt/ARCHIVED"

main_inventory="$(mktemp)"
ttt_inventory="$(mktemp)"
trap 'rm -f -- "$main_inventory" "$ttt_inventory"' EXIT
find "$main_attempt" -printf '%y|%s|%TY-%Tm-%TdT%TH:%TM:%TS|%p\n' | sort > "$main_inventory"
find "$ttt_attempt" -printf '%y|%s|%TY-%Tm-%TdT%TH:%TM:%TS|%p\n' | sort > "$ttt_inventory"
mv -- "$main_inventory" "$main_attempt/archive_inventory.txt"
mv -- "$ttt_inventory" "$ttt_attempt/archive_inventory.txt"
trap - EXIT

test ! -e "$main_phase"
test ! -e "$run_dir/phases/merged_serve_smoke"
test ! -e "$run_dir/training_source_snapshot"
test ! -e "$run_dir/checkpoints/training_smoke"
test ! -e "$ttt_phase"
test ! -e "$ttt_dir/training_source_snapshot"
test ! -e "$ttt_dir/training"
test ! -e "$ttt_dir/checkpoints"
test ! -e "$ttt_dir/exports"
test ! -e "$ttt_dir/after_rollouts"

printf 'recovery_ready_utc=%s\nmain_failed_job_id=%s\nttt_failed_job_id=%s\nrepair_job_id=%s\n' \
  "$(date -Is -u)" "$main_job_id" "$ttt_job_id" "$repair_job_id" \
  > "$run_dir/preflight/TRAINING_RAY_RELOCATION_RECOVERY_READY"
echo "archived Ray-wrapper failures; canonical recovery slots are ready"
