#!/usr/bin/env bash
# Host-side, resumable driver for the six independent executor-update baselines.
# It intentionally remains outside every container; all sbatch calls originate
# here.  Existing proposer campaigns are never submitted by this driver.
set -euo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh

SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
STATE_ROOT="${STATE_ROOT:-$RUN_ROOT/self_adapt_harness/ttt_discover_12h}"
BASE="$MODEL_ROOT/base/Qwen3.5-9B/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
UPDATES="${UPDATES:-3}"
K="${K:-16}"
MODEL_TAG="${MODEL_TAG:-ttt12b}"
PLOT_LAYOUT="${PLOT_LAYOUT:-2x3}"
PLOT_PREFIX="${PLOT_PREFIX:-$SAH/papers/figures/score_compute_curves_12h}"
PLOT_AHC039_STATE_ROOT="${PLOT_AHC039_STATE_ROOT:-$RUN_ROOT/self_adapt_harness/ttt_discover_clean20}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(( $(date +%s) + 39600 ))}"  # 11h default
MIN_START_SECONDS="${MIN_START_SECONDS:-5400}"  # leave time to finish + plot
PARTIAL_AFTER_SECONDS="${PARTIAL_AFTER_SECONDS:-4200}"
FIXED_LAUNCHED_BATCH="${FIXED_LAUNCHED_BATCH:-0}"
MIN_TRAIN_USABLE="${MIN_TRAIN_USABLE:-8}"
PARTIAL_MIN_USABLE="${PARTIAL_MIN_USABLE:-}"
[ -n "$PARTIAL_MIN_USABLE" ] || {
  if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
    PARTIAL_MIN_USABLE="$MIN_TRAIN_USABLE"
  else
    PARTIAL_MIN_USABLE="$K"
  fi
}
AHC_SUPPLEMENT_K="${AHC_SUPPLEMENT_K:-0}"
MAX_SUPPLEMENT_LAUNCHES="${MAX_SUPPLEMENT_LAUNCHES:-8}"
TASK_FILTER="${TASK_FILTER:-}"
DRIVER_LOCK_SUFFIX="${DRIVER_LOCK_SUFFIX:-}"
mkdir -p "$STATE_ROOT"
log(){ echo "[$(date -Is)] [ttt12-driver] $*"; }

# A terminated foreground shell can leave its six task monitors adopted by
# init.  Serialize the driver so a later resume cannot race those monitors and
# submit duplicate train/merge/eval chains.
DRIVER_LOCK="$STATE_ROOT/driver_${MODEL_TAG}${DRIVER_LOCK_SUFFIX}.lock"
exec 9>"$DRIVER_LOCK"
if ! flock -n 9; then
  log "another ${MODEL_TAG} driver still owns $DRIVER_LOCK; exiting"
  exit 0
fi
printf 'pid=%s started=%s\n' "$$" "$(date -Is)" >&9

case "${TASK_SET:-math6}" in
  math6)
    TASKS=(
      eft__math__erdos_min_overlap
      eft__math__circle_packing
      eft__math__hadamard_maximal_det
      eft__math__first_autocorr_ineq
      eft__math__second_autocorr_ineq
      eft__ahc_simpletes__ahc039
    )
    TAGS=(erdos circle hadamard ac1 ac2 ahc039)
    ;;
  sota4)
    # ahc039 already has a clean20 executor-update state; this set fills the
    # four missing tasks needed by the five-panel reward-routing figure.
    TASKS=(
      eft__ahc_simpletes__ahc058
      adrs__eplb
      adrs__prism
      adrs__llm_sql
    )
    TAGS=(ahc058 eplb prism llmsql)
    ;;
  sota7extra)
    # Extra priority tasks added to the canonical reward-route comparison.
    # Hadamard's older K=32/40 executor run remains sensitivity-only; this set
    # gives Hadamard and Transaction the same K=8 cadence as SOTA5.
    TASKS=(
      eft__math__hadamard_maximal_det
      adrs__txn_scheduling
    )
    TAGS=(hadamard txnsched)
    ;;
  curve5)
    TASKS=(
      eft__math__erdos_min_overlap
      eft__math__second_autocorr_ineq
      eft__math__hadamard_maximal_det
      eft__ahc_simpletes__ahc039
      adrs__eplb
    )
    TAGS=(erdos ac2 hadamard ahc039 eplb)
    ;;
  curve4)
    TASKS=(
      eft__math__erdos_min_overlap
      eft__math__second_autocorr_ineq
      eft__math__hadamard_maximal_det
      adrs__eplb
    )
    TAGS=(erdos ac2 hadamard eplb)
    ;;
  *)
    log "unknown TASK_SET=${TASK_SET}; expected math6, sota4, sota7extra, curve5, or curve4"
    exit 2
    ;;
esac

slurm_up(){ timeout 20s scontrol ping 2>/dev/null | grep -q 'UP'; }

terminal_state(){
  local job="$1" state
  state=$(timeout 20s sacct -j "$job" -X -n -o State 2>/dev/null | head -1 | xargs || true)
  case "$state" in
    COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL*) echo "$state" ;;
    *) echo "" ;;
  esac
}

count_usable(){
  python3 - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

out, task = Path(sys.argv[1]), sys.argv[2]
usable = 0
for kdir in out.glob("k*"):
    if not kdir.is_dir() or not kdir.name[1:].isdigit():
        continue
    summaries = sorted(kdir.glob("*/summary.json"))
    if not summaries:
        continue
    try:
        payload = json.loads(summaries[-1].read_text())
    except Exception:
        continue
    payload = payload if isinstance(payload, list) else [payload]
    row = next((item for item in payload if item.get("task_id") == task), None)
    if row and row.get("best_score") is not None and row.get("best_program"):
        usable += 1
print(usable)
PY
}

count_launched(){
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
out = Path(sys.argv[1])
print(sum(1 for p in out.glob("k*.log") if p.stem[1:].isdigit()))
PY
}

collect_eval_manifest(){
  # Keep every manual-collection path on the same H2 hash/seed contract as the
  # normal eval worker.  Callers may additionally pass --slurm-job-id.
  local -a provenance_args
  provenance_args=(--seed-base "${H2_SEED_BASE:-200000}")
  [ -z "${FIXED_H2_SHA256:-}" ] || \
    provenance_args+=(--expect-h2-sha256 "$FIXED_H2_SHA256")
  python3 "$SAH/scripts/runtime/collect_ttt_eval_manifest.py" \
    "$@" "${provenance_args[@]}"
}

collect_fixed_terminal_manifest(){
  # A terminal fixed-budget job may be killed by its scheduler lease before
  # the worker writes eval_manifest.json.  Once all K log files exist, those K
  # trajectories have been launched and are charged.  Materialize the partial
  # manifest without top-up; a low-usable batch skips the update downstream.
  local out_dir="$1" task="$2" checkpoint="$3" worker_rc="$4" reason="$5"
  local slurm_job_id="${6:-}"
  local launched usable
  launched=$(count_launched "$out_dir")
  usable=$(count_usable "$out_dir" "$task")
  [ "$launched" -eq "$K" ] || {
    log "fixed terminal batch launched $launched/$K trajectories; lineage is incomplete"
    return 1
  }
  local -a job_args=()
  [ -z "$slurm_job_id" ] || job_args+=(--slurm-job-id "$slurm_job_id")
  collect_eval_manifest \
    --out-dir "$out_dir" --task "$task" --checkpoint "$checkpoint" \
    --target "$K" --launched "$launched" --worker-rc "$worker_rc" \
    --reason "${reason}_usable_${usable}_target_${K}_launched_${launched}" \
    --min-usable 0 "${job_args[@]}" >/dev/null
}

wait_for_eval(){
  local job="$1" manifest="$2" task="$3" checkpoint="$4"
  local supplement_job="${5:-}" charged="${6:-$K}" state
  local out_dir active_since=0 now usable launched_actual reason worker_rc
  out_dir=$(dirname "$manifest")
  while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do
    [ -s "$manifest" ] && return 0
    if [ "$active_since" -eq 0 ]; then
      if [ -s "$out_dir/worker_started_epoch" ]; then
        active_since=$(cat "$out_dir/worker_started_epoch" 2>/dev/null || echo 0)
      elif [ -e "$out_dir/pip.log" ]; then
        active_since=$(stat -c %Y "$out_dir/pip.log" 2>/dev/null || echo 0)
      fi
    fi
    now=$(date +%s)
    usable=$(count_usable "$out_dir" "$task")
    if [ "$usable" -ge "$K" ]; then
      launched_actual=$(python3 - "$out_dir" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
print(sum(1 for p in out.glob("k*.log") if p.stem[1:].isdigit()))
PY
)
      if [ "$launched_actual" -gt "$charged" ]; then
        log "eval job $job: counted $launched_actual launches above capacity $charged"
        return 1
      fi
      scancel "$job" 2>/dev/null || true
      [ -n "$supplement_job" ] && scancel "$supplement_job" 2>/dev/null || true
      collect_eval_manifest \
        --out-dir "$out_dir" --task "$task" --checkpoint "$checkpoint" \
        --target "$K" --launched "$launched_actual" --worker-rc 0 \
        --reason "usable_target_met_${usable}_target_${K}_launched_${launched_actual}" \
        --min-usable "$K" --slurm-job-id "$job" >/dev/null
      return 0
    fi
    if [ "$active_since" -gt 0 ] && [ $((now - active_since)) -ge "$PARTIAL_AFTER_SECONDS" ]; then
      if [ "$usable" -ge "$PARTIAL_MIN_USABLE" ]; then
        log "eval job $job: cutting tail stragglers after $((now-active_since))s with $usable/$K usable"
        scancel "$job" 2>/dev/null || true
        [ -n "$supplement_job" ] && scancel "$supplement_job" 2>/dev/null || true
        sleep 15
        [ -s "$manifest" ] && return 0
        usable=$(count_usable "$out_dir" "$task")
        launched_actual=$(python3 - "$out_dir" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
print(sum(1 for p in out.glob("k*.log")
          if p.stem[1:].isdigit()))
PY
)
        [ "$launched_actual" -ge "$usable" ] || launched_actual="$usable"
        if [ "$launched_actual" -gt "$charged" ]; then
          log "eval job $job: counted $launched_actual launches above capacity $charged"
          return 1
        fi
        if [ "$usable" -ge "$K" ]; then
          reason="complete_manual_collection_${usable}_target_${K}_launched_${launched_actual}"
          worker_rc=0
        else
          reason="straggler_cutoff_${usable}_target_${K}_launched_${launched_actual}"
          worker_rc=124
        fi
        collect_eval_manifest \
          --out-dir "$out_dir" --task "$task" --checkpoint "$checkpoint" \
          --target "$K" --launched "$launched_actual" --worker-rc "$worker_rc" \
          --reason "$reason" --min-usable "$PARTIAL_MIN_USABLE" \
          --slurm-job-id "$job" >/dev/null
        return 0
      fi
    fi
    if slurm_up; then
      state=$(terminal_state "$job")
      case "$state" in
        COMPLETED)
          sleep 10
          [ -s "$manifest" ] && return 0
          if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
            collect_fixed_terminal_manifest "$out_dir" "$task" "$checkpoint" 0 \
              completed_manual_collection "$job" && return 0
            return 1
          fi
          usable=$(count_usable "$out_dir" "$task")
          launched_actual=$(python3 - "$out_dir" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
print(sum(1 for p in out.glob("k*.log") if p.stem[1:].isdigit()))
PY
)
          if [ "$usable" -ge "$PARTIAL_MIN_USABLE" ]; then
            collect_eval_manifest \
              --out-dir "$out_dir" --task "$task" --checkpoint "$checkpoint" \
              --target "$K" --launched "$launched_actual" --worker-rc 0 \
              --reason "completed_manual_collection_${usable}_target_${K}_launched_${launched_actual}" \
              --min-usable "$PARTIAL_MIN_USABLE" --slurm-job-id "$job" >/dev/null
            return 0
          fi
          return 1
          ;;
        "") sleep 30 ;;
        *)
          log "eval job $job terminated as $state"
          if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
            collect_fixed_terminal_manifest "$out_dir" "$task" "$checkpoint" 124 \
              "terminal_${state}" "$job" && return 0
          fi
          return 1
          ;;
      esac
    else
      sleep 30
    fi
  done
  log "deadline reached waiting for eval job $job"
  if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
    collect_fixed_terminal_manifest "$out_dir" "$task" "$checkpoint" 124 \
      driver_deadline "$job" && return 0
  fi
  return 1
}

ensure_target_usable(){
  # A failed trajectory is charged, not silently replaced.  Keep launching
  # uniquely indexed supplements until the update has K usable trajectories,
  # so every canonical replay contains enough distinct rows for two GBS=4
  # optimizer boundaries.  Partial manifests are retained as provenance.
  local out_dir="$1" task="$2" checkpoint="$3" parent_file="${4:-}"
  local usable launched capacity batch job attempt manifest archive marker_archive pip_archive
  local -a sched
  manifest="$out_dir/eval_manifest.json"
  if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
    # Inference-16 spends exactly K=16 launched H2 trajectories.  It never
    # erases failures by adding replacement trajectories: failures remain on
    # the x-axis and training uses the largest four-way-safe usable subset.
    [ -s "$manifest" ] || {
      log "fixed batch missing terminal manifest: $manifest"
      return 1
    }
    usable=$(count_usable "$out_dir" "$task")
    launched=$(python3 - "$out_dir" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
print(sum(1 for p in out.glob("k*.log") if p.stem[1:].isdigit()))
PY
)
    if [ "$launched" -ne "$K" ]; then
      log "fixed batch launched $launched trajectories, expected exactly $K"
      return 1
    fi
    if [ "$usable" -lt "$MIN_TRAIN_USABLE" ]; then
      log "fixed batch accepted with $usable usable rows; executor update will be skipped (minimum=$MIN_TRAIN_USABLE)"
    else
      log "fixed batch accepted: launched=$launched usable=$usable (no top-up)"
    fi
    return 0
  fi
  # Transaction, AHC, and Hadamard trajectories can legitimately reach the
  # short-QoS two-hour boundary even though each executor call still has the
  # same 420-second timeout and MAX_EVALS=20 cap.  A longer scheduler lease
  # prevents partial batches; it does not change the logical method budget,
  # and actual GPU-hours remain charged by sacct.
  if [[ "$task" == *ahc* || "$task" == "adrs__txn_scheduling" ||
        "$task" == "eft__math__hadamard_maximal_det" ]]; then
    sched=(--qos=normal --time=04:00:00)
  else
    sched=(--qos=short --time=02:00:00)
  fi
  while :; do
    usable=$(count_usable "$out_dir" "$task")
    launched=$(python3 - "$out_dir" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
print(sum(1 for p in out.glob("k*.log") if p.stem[1:].isdigit()))
PY
)
    if [ "$usable" -ge "$K" ]; then
      collect_eval_manifest \
        --out-dir "$out_dir" --task "$task" --checkpoint "$checkpoint" \
        --target "$K" --launched "$launched" --worker-rc 0 \
        --reason "usable_target_met_${usable}_target_${K}_launched_${launched}" \
        --min-usable "$K" >/dev/null
      return 0
    fi
    capacity=$((K + MAX_SUPPLEMENT_LAUNCHES - launched))
    if [ "$capacity" -le 0 ]; then
      log "eval top-up exhausted after $launched launches with $usable/$K usable"
      return 1
    fi
    batch=$((K - usable))
    [ "$batch" -le "$capacity" ] || batch="$capacity"
    if [ -s "$manifest" ]; then
      archive="$out_dir/eval_manifest.partial_l${launched}_u${usable}.json"
      [ ! -e "$archive" ] || archive="${archive}.$(date +%s)"
      mv "$manifest" "$archive"
    fi
    # ``wait_for_eval`` must not interpret the original batch's timestamp as
    # the new supplement's runtime while the supplement is still pending.
    if [ -e "$out_dir/worker_started_epoch" ]; then
      marker_archive="$out_dir/worker_started_epoch.before_topup_l${launched}_u${usable}"
      [ ! -e "$marker_archive" ] || marker_archive="${marker_archive}.$(date +%s)"
      mv "$out_dir/worker_started_epoch" "$marker_archive"
    fi
    if [ -e "$out_dir/pip.log" ]; then
      pip_archive="$out_dir/pip.before_topup_l${launched}_u${usable}.log"
      [ ! -e "$pip_archive" ] || pip_archive="${pip_archive}.$(date +%s)"
      mv "$out_dir/pip.log" "$pip_archive"
    fi
    job=""
    for attempt in $(seq 1 12); do
      job=$(cd "$SAH" && sbatch --parsable "${sched[@]}" \
        --export="ALL,TASK=$task,TTT_CKPT=$checkpoint,OUT_DIR=$out_dir,K=$batch,K_START=$launched,MAX_EVALS=20,EVAL_TIMEOUT=420,PARENT_FILE=$parent_file,WRITE_MANIFEST=0" \
        scripts/ttt_executor_eval.sbatch 2>/dev/null || true)
      [[ "$job" =~ ^[0-9]+$ ]] && break
      log "eval top-up submission attempt $attempt failed; retrying"
      sleep 15
    done
    [[ "$job" =~ ^[0-9]+$ ]] || return 1
    log "eval top-up job=$job k_start=$launched k=$batch (currently $usable/$K usable)"
    wait_for_eval "$job" "$manifest" "$task" "$checkpoint" "" \
      "$((launched + batch))" || true
  done
}

run_task(){
  local task="$1" tag="$2" sd
  sd="$STATE_ROOT/$tag"
  mkdir -p "$sd"
  local update replay prev submit_out tj mj ej merged lora eval_dir parent_file parent_id job_file charged kl
  local train_rows train_allowed eval_checkpoint previous_checkpoint train_skipped
  local base_eval_dir base_job_file supplement_ej charged_total write_manifest
  local -a eval_sched
  if [[ "$task" == *ahc* || "$task" == "adrs__txn_scheduling" ||
        "$task" == "eft__math__hadamard_maximal_det" ]]; then
    eval_sched=(--qos=normal --time=04:00:00)
    charged_total=$((K + AHC_SUPPLEMENT_K))
    [ "$AHC_SUPPLEMENT_K" -gt 0 ] && write_manifest=0 || write_manifest=1
  else
    eval_sched=(--qos=short --time=02:00:00)
    charged_total="$K"
    write_manifest=1
  fi

  # A clean comparison needs a full-budget base-model batch under the exact
  # same MAX_EVALS=20 harness as every post-update point.  Older cached TTT
  # batches used only ten evaluator calls and are intentionally not imported.
  if [ ! -s "$sd/prepare_step00.json" ]; then
    base_eval_dir="$sd/eval_${MODEL_TAG}_u0"
    base_job_file="$sd/jobs_${MODEL_TAG}_u0.env"
    if [ ! -s "$base_eval_dir/eval_manifest.json" ]; then
      if [ -s "$base_job_file" ]; then
        # shellcheck disable=SC1090
        source "$base_job_file"
        ej="$EVAL_JOB"
        supplement_ej="${SUPP_EVAL_JOB:-}"
        log "$tag update0: resuming base eval job $ej"
      else
        ej=""
        for attempt in $(seq 1 12); do
          ej=$(cd "$SAH" && sbatch --parsable "${eval_sched[@]}" \
            --export="ALL,TASK=$task,TTT_CKPT=$BASE,OUT_DIR=$base_eval_dir,K=$K,H2_BATCH_INDEX=0,MAX_EVALS=20,EVAL_TIMEOUT=420,WRITE_MANIFEST=$write_manifest" \
            scripts/ttt_executor_eval.sbatch 2>/dev/null || true)
          [[ "$ej" =~ ^[0-9]+$ ]] && break
          log "$tag update0: base eval submission attempt $attempt failed; retrying"
          sleep 15
        done
        [[ "$ej" =~ ^[0-9]+$ ]] || { log "$tag update0: bad eval id $ej"; return 1; }
        supplement_ej=""
        if [ "$AHC_SUPPLEMENT_K" -gt 0 ] && [[ "$task" == *ahc* ]]; then
          supplement_ej=$(cd "$SAH" && sbatch --parsable "${eval_sched[@]}" \
            --export="ALL,TASK=$task,TTT_CKPT=$BASE,OUT_DIR=$base_eval_dir,K=$AHC_SUPPLEMENT_K,K_START=$K,H2_BATCH_INDEX=0,MAX_EVALS=20,EVAL_TIMEOUT=420,WRITE_MANIFEST=0" \
            scripts/ttt_executor_eval.sbatch)
          [[ "$supplement_ej" =~ ^[0-9]+$ ]] || { log "$tag update0: bad supplemental id"; return 1; }
        fi
        printf 'EVAL_JOB=%q\nSUPP_EVAL_JOB=%q\nCHECKPOINT=%q\n' \
          "$ej" "$supplement_ej" "$BASE" > "$base_job_file"
        log "$tag update0: base eval=$ej supplement=${supplement_ej:-none}"
      fi
      wait_for_eval "$ej" "$base_eval_dir/eval_manifest.json" "$task" "$BASE" \
        "$supplement_ej" "$charged_total" || return 1
    fi
    ensure_target_usable "$base_eval_dir" "$task" "$BASE" "" || return 1
    charged=$(python3 - "$base_eval_dir/eval_manifest.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))["launched"]))
PY
)
    python3 "$SAH/scripts/runtime/ttt_discover_prepare.py" \
      --task "$task" --round-dir "$base_eval_dir" --state-dir "$sd" \
      --step 0 --launched "$charged" --checkpoint "$BASE" --parent-id root \
      --max-train-rows "$K" --min-train-rows "$MIN_TRAIN_USABLE" \
      > "$sd/prepare_step00.log"
    log "$tag update0: prepared full-budget base replay and curve point"
    if [ -n "${POST_BATCH_HOOK:-}" ]; then
      ROUND_DIR="$base_eval_dir" BATCH_INDEX=0 ROUTE=executor \
        bash "$POST_BATCH_HOOK" || log "$tag update0: post-batch hook failed (non-fatal)"
    fi
  fi

  for update in $(seq 1 "$UPDATES"); do
    [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ] || { log "$tag: deadline"; return 0; }
    replay="$sd/replay_step$(printf '%02d' "$update").jsonl"
    train_rows=0
    [ -s "$replay" ] && train_rows=$(wc -l < "$replay")
    train_allowed=1
    if [ "$FIXED_LAUNCHED_BATCH" = 1 ] && [ "$train_rows" -lt "$MIN_TRAIN_USABLE" ]; then
      train_allowed=0
      log "$tag batch$update: previous batch has $train_rows train rows; keeping executor weights unchanged"
    elif [ ! -s "$replay" ]; then
      log "$tag: missing replay $replay"
      return 1
    fi
    parent_file="$sd/parent_step$(printf '%02d' "$update").json"
    [ -s "$parent_file" ] || { log "$tag: missing parent $parent_file"; return 1; }
    eval_dir="$sd/eval_${MODEL_TAG}_u$update"
    job_file="$sd/jobs_${MODEL_TAG}_u$update.env"
    merged="$MODEL_ROOT/exports/self_adapt_harness/${MODEL_TAG}_${tag}_u${update}"
    lora="$MODEL_ROOT/checkpoints/self_adapt_harness/${MODEL_TAG}_${tag}_u${update}"
    eval_checkpoint="$merged"

    if [ -s "$eval_dir/eval_manifest.json" ]; then
      log "$tag update$update: eval already complete"
      eval_checkpoint=$(python3 - "$eval_dir/eval_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["checkpoint"])
PY
)
    else
      if [ -s "$job_file" ]; then
        TRAIN_SKIPPED=0 CHECKPOINT="" MERGED="$merged" SUPP_EVAL_JOB=""
        # shellcheck disable=SC1090
        source "$job_file"
        ej="$EVAL_JOB"
        supplement_ej="${SUPP_EVAL_JOB:-}"
        eval_checkpoint="${CHECKPOINT:-${MERGED:-$merged}}"
        log "$tag update$update: resuming eval job $ej (train_skipped=${TRAIN_SKIPPED:-0})"
      else
        if [ $((DEADLINE_EPOCH - $(date +%s))) -lt "$MIN_START_SECONDS" ]; then
          log "$tag: not starting update$update inside the ${MIN_START_SECONDS}s completion guard"
          return 0
        fi
        tj=""; mj=""; kl="skipped"; train_skipped=0
        if [ "$train_allowed" = 1 ]; then
          prev=""
          if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
            previous_checkpoint=$(python3 - "$sd/curve.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(rows[-1]["checkpoint"])
PY
)
            case "$previous_checkpoint" in
              "$MODEL_ROOT/exports/self_adapt_harness/${MODEL_TAG}_${tag}_u"*)
                prev="$MODEL_ROOT/checkpoints/self_adapt_harness/$(basename "$previous_checkpoint")"
                [ -d "$prev" ] || { log "$tag update$update: missing prior LoRA $prev"; return 1; }
                ;;
            esac
          elif [ "$update" -gt 1 ]; then
            prev="$MODEL_ROOT/checkpoints/self_adapt_harness/${MODEL_TAG}_${tag}_u$((update-1))"
          fi
          submit_out=$(TTT_PREFIX="$MODEL_TAG" bash "$SAH/scripts/submit_ttt_executor_update.sh" \
            "$tag" "$update" "$replay" "$prev")
          tj=$(echo "$submit_out" | sed -n 's/.*train_job=\([0-9][0-9]*\).*/\1/p' | head -1)
          mj=$(echo "$submit_out" | sed -n 's/.*merge_job=\([0-9][0-9]*\).*/\1/p' | head -1)
          kl=$(echo "$submit_out" | sed -n 's/.*kl_coef=\([^ ]*\).*/\1/p' | head -1)
          [[ "$tj" =~ ^[0-9]+$ && "$mj" =~ ^[0-9]+$ ]] || {
            log "$tag update$update: could not parse train/merge ids: $submit_out"; return 1; }
          eval_checkpoint="$merged"
        else
          train_skipped=1
          eval_checkpoint=$(python3 - "$sd/curve.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
print(rows[-1]["checkpoint"])
PY
)
        fi
        ej=""
        for attempt in $(seq 1 12); do
          dependency=()
          [ -n "$mj" ] && dependency+=(--dependency="afterok:$mj")
          ej=$(cd "$SAH" && sbatch --parsable "${eval_sched[@]}" "${dependency[@]}" \
            --export="ALL,TASK=$task,TTT_CKPT=$eval_checkpoint,OUT_DIR=$eval_dir,K=$K,H2_BATCH_INDEX=$update,MAX_EVALS=20,EVAL_TIMEOUT=420,PARENT_FILE=$parent_file,WRITE_MANIFEST=$write_manifest" \
            scripts/ttt_executor_eval.sbatch 2>/dev/null || true)
          [[ "$ej" =~ ^[0-9]+$ ]] && break
          log "$tag update$update: eval submission attempt $attempt failed; retrying"
          sleep 15
        done
        [[ "$ej" =~ ^[0-9]+$ ]] || { log "$tag update$update: bad eval id $ej"; return 1; }
        supplement_ej=""
        if [ "$AHC_SUPPLEMENT_K" -gt 0 ] && [[ "$task" == *ahc* ]]; then
          supplement_ej=$(cd "$SAH" && sbatch --parsable "${eval_sched[@]}" \
            --dependency="afterok:$mj" \
            --export="ALL,TASK=$task,TTT_CKPT=$eval_checkpoint,OUT_DIR=$eval_dir,K=$AHC_SUPPLEMENT_K,K_START=$K,H2_BATCH_INDEX=$update,MAX_EVALS=20,EVAL_TIMEOUT=420,PARENT_FILE=$parent_file,WRITE_MANIFEST=0" \
            scripts/ttt_executor_eval.sbatch)
          [[ "$supplement_ej" =~ ^[0-9]+$ ]] || { log "$tag update$update: bad supplemental id"; return 1; }
        fi
        printf 'TRAIN_SKIPPED=%q\nTRAIN_JOB=%q\nMERGE_JOB=%q\nEVAL_JOB=%q\nSUPP_EVAL_JOB=%q\nKL_COEF=%q\nLORA=%q\nMERGED=%q\nCHECKPOINT=%q\n' \
          "$train_skipped" "$tj" "$mj" "$ej" "$supplement_ej" "${kl:-unknown}" "$lora" "$merged" "$eval_checkpoint" > "$job_file"
        log "$tag update$update: train=${tj:-skipped} merge=${mj:-skipped} eval=$ej supplement=${supplement_ej:-none} kl=${kl:-skipped} checkpoint=$(basename "$eval_checkpoint")"
      fi
      wait_for_eval "$ej" "$eval_dir/eval_manifest.json" "$task" "$eval_checkpoint" \
        "$supplement_ej" "$charged_total" || return 1
    fi
    ensure_target_usable "$eval_dir" "$task" "$eval_checkpoint" "$parent_file" || return 1

    if [ ! -s "$sd/prepare_step$(printf '%02d' "$update").json" ]; then
      parent_id=$(python3 - "$parent_file" "$task" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))[sys.argv[2]]["state_id"])
PY
)
      charged=$(python3 - "$eval_dir/eval_manifest.json" <<'PY'
import json,sys
print(int(json.load(open(sys.argv[1]))["launched"]))
PY
)
      python3 "$SAH/scripts/runtime/ttt_discover_prepare.py" \
        --task "$task" --round-dir "$eval_dir" --state-dir "$sd" \
        --step "$update" --launched "$charged" --checkpoint "$eval_checkpoint" \
        --parent-id "$parent_id" --max-train-rows "$K" \
        --min-train-rows "$MIN_TRAIN_USABLE" \
        > "$sd/prepare_step$(printf '%02d' "$update").log"
      log "$tag update$update: prepared next replay and curve point"
      if [ -n "${POST_BATCH_HOOK:-}" ]; then
        ROUND_DIR="$eval_dir" BATCH_INDEX="$update" ROUTE=executor \
          bash "$POST_BATCH_HOOK" || log "$tag update$update: post-batch hook failed (non-fatal)"
      fi
      if [ "$PLOT_LAYOUT" != "none" ]; then
        TTT_STATE_ROOT="$STATE_ROOT" TTT_AHC039_STATE_ROOT="$PLOT_AHC039_STATE_ROOT" \
          TTT_EPLB_STATE_ROOT="$STATE_ROOT" TTT_MODEL_TAG="$MODEL_TAG" \
          TTT_BATCH_K="$K" \
          MPLCONFIGDIR=/tmp/mpl-ttt12 PYTHONPATH=/tmp/reward-route-pydeps \
          python3 "$SAH/scripts/analysis/figures/reward_route_12h.py" \
          --layout "$PLOT_LAYOUT" --out-prefix "$PLOT_PREFIX" \
          > "$STATE_ROOT/plot.log" 2>&1 || true
      fi
    fi
  done
  log "$tag: completed $((UPDATES + 1)) fixed executor batches ($UPDATES update opportunities)"
}

if [ "$FIXED_LAUNCHED_BATCH" = 1 ]; then
  [ "$K" -eq 16 ] || { log "fixed inference comparison requires K=16"; exit 2; }
  [ "$AHC_SUPPLEMENT_K" -eq 0 ] || { log "fixed comparison forbids AHC supplements"; exit 2; }
  [ "$MAX_SUPPLEMENT_LAUNCHES" -eq 0 ] || { log "fixed comparison forbids top-ups"; exit 2; }
  [ "${TTT_MATCH_PROPOSER_TRAIN:-0}" = 1 ] || {
    log "fixed comparison requires TTT_MATCH_PROPOSER_TRAIN=1"; exit 2; }
fi
log "starting ${#TASKS[@]} task pipelines; set=${TASK_SET:-math6} updates=$UPDATES K=$K fixed_launched=$FIXED_LAUNCHED_BATCH deadline=$DEADLINE_EPOCH"
pids=()
for i in "${!TASKS[@]}"; do
  [ -n "$TASK_FILTER" ] && [ "${TAGS[$i]}" != "$TASK_FILTER" ] && continue
  mkdir -p "$STATE_ROOT/${TAGS[$i]}"
  # Preserve earlier resumable-driver evidence; overwriting this log loses the
  # update/job chain even though the checkpoints and manifests survive.
  run_task "${TASKS[$i]}" "${TAGS[$i]}" >> "$STATE_ROOT/${TAGS[$i]}/driver.log" 2>&1 &
  pids+=("$!")
done
[ "${#pids[@]}" -gt 0 ] || { log "no task matched TASK_FILTER=$TASK_FILTER"; exit 2; }
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$PLOT_LAYOUT" != "none" ]; then
  TTT_STATE_ROOT="$STATE_ROOT" TTT_AHC039_STATE_ROOT="$PLOT_AHC039_STATE_ROOT" \
    TTT_EPLB_STATE_ROOT="$STATE_ROOT" TTT_MODEL_TAG="$MODEL_TAG" \
    TTT_BATCH_K="$K" \
    MPLCONFIGDIR=/tmp/mpl-ttt12 PYTHONPATH=/tmp/reward-route-pydeps \
    python3 "$SAH/scripts/analysis/figures/reward_route_12h.py" \
    --layout "$PLOT_LAYOUT" --out-prefix "$PLOT_PREFIX" \
    > "$STATE_ROOT/plot.log" 2>&1 || rc=1
fi
log "driver finished rc=$rc"
exit "$rc"
