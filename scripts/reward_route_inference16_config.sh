#!/usr/bin/env bash
# Authoritative configuration for the four-task inference-trajectory comparison.
# Sourcing this file has no side effects and never submits work.

RR_PROTOCOL_ID="reward-route-inference16-v1"
RR_TASK_TAGS=(erdos ac2 hadamard eplb)

RR_H1_TRAJECTORIES=8
RR_H2_TRAJECTORIES_H1_ROUTE=8
RR_H2_TRAJECTORIES_EXECUTOR_ROUTE=16
RR_TRAJECTORIES_PER_ROUND=16
RR_ROUNDS=19
RR_WEIGHT_UPDATES=18
RR_EXECUTOR_POST_UPDATE_EVALS=18  # base batch + 18 post-update batches = 19
RR_SHARED_ANCHOR_TRAJECTORIES=1
RR_FINAL_X=$((RR_SHARED_ANCHOR_TRAJECTORIES + RR_ROUNDS * RR_TRAJECTORIES_PER_ROUND))

RR_MAX_EVALS=20
RR_EVAL_TIMEOUT=420
RR_ROLLOUT_WALL_TIMEOUT=10800
RR_FORCE_TOOL_FRAC=0.25
RR_LOGICAL_SEED_BASE=1000
RR_H2_SEED_BASE=200000
RR_H1_MAX_ITERATIONS=24
RR_PROGRAM_RATCHET_MODE=strict_single
RR_MIN_TRAINABLE_H1_ROWS=4

RR_LORA_RANK=64
RR_LORA_ALPHA=128
RR_EPOCHS=3
RR_LR=3e-5
RR_KL=0.05
RR_PROPOSER_GLOBAL_BATCH=8
RR_EXECUTOR_MAX_GLOBAL_BATCH=16
RR_MICRO_BATCH=1

rr_task_id(){
  case "$1" in
    erdos) echo eft__math__erdos_min_overlap ;;
    ac2) echo eft__math__second_autocorr_ineq ;;
    hadamard) echo eft__math__hadamard_maximal_det ;;
    eplb) echo adrs__eplb ;;
    *) echo "unknown inference16 task tag: $1" >&2; return 2 ;;
  esac
}

rr_task_index(){
  case "$1" in
    erdos) echo 0 ;; ac2) echo 1 ;; hadamard) echo 2 ;;
    eplb) echo 3 ;;
    *) return 2 ;;
  esac
}

rr_round_base(){
  local method="$1" tag="$2" index offset
  index=$(rr_task_index "$tag") || return
  case "$method" in
    proposer) offset=2000 ;;
    context) offset=3000 ;;
    *) echo "round base is defined only for proposer/context" >&2; return 2 ;;
  esac
  echo $((offset + index * 100))
}

rr_assert_config(){
  [ "$RR_H1_TRAJECTORIES" -eq 8 ]
  [ "$RR_H2_TRAJECTORIES_H1_ROUTE" -eq 8 ]
  [ "$RR_H2_TRAJECTORIES_EXECUTOR_ROUTE" -eq 16 ]
  [ $((RR_H1_TRAJECTORIES + RR_H2_TRAJECTORIES_H1_ROUTE)) -eq "$RR_TRAJECTORIES_PER_ROUND" ]
  [ "$RR_H2_TRAJECTORIES_EXECUTOR_ROUTE" -eq "$RR_TRAJECTORIES_PER_ROUND" ]
  [ "$RR_FINAL_X" -eq 305 ]
  [ "$RR_WEIGHT_UPDATES" -eq $((RR_ROUNDS - 1)) ]
  [ "$RR_EXECUTOR_POST_UPDATE_EVALS" -eq "$RR_WEIGHT_UPDATES" ]
  [ "$RR_LORA_RANK" -eq 64 ]
  [ "$RR_LORA_ALPHA" -eq 128 ]
  [ "$RR_EPOCHS" -eq 3 ]
  [ "$RR_H1_MAX_ITERATIONS" -eq 24 ]
  [ "$RR_PROGRAM_RATCHET_MODE" = strict_single ]
  [ "$RR_MIN_TRAINABLE_H1_ROWS" -eq 4 ]
  grep -q '^max_iterations: 24$' \
    "$CODE_ROOT/self_adapt_harness/src/outer/harness/agent.yaml"
}
