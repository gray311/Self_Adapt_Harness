#!/usr/bin/env bash
# One-round SCRATCH smoke test of the fixed pipeline (file-native H1, tool
# gates, note guard, task-text pinning, strict_single ratchet).
#
# Deliberately NOT the canonical inference16 driver: output goes to the
# disposable namespace outer-smoke-fixv2/, so a failure burns nothing.
#
#   bash scripts/smoke_test_fixed_pipeline.sh          # submit
#   OUT_TAG=smoke-fixv2b bash scripts/...              # fresh namespace
set -euo pipefail
cd "$(dirname "$0")/.."

export ROUND_ID="${ROUND_ID:-1}"
export TASKS="${TASKS:-eft__math__second_autocorr_ineq}"
export K="${K:-4}"
export MAX_EVALS="${MAX_EVALS:-5}"
export OUT_TAG="${OUT_TAG:-smoke-fixv2}"
# exercise the new enforcement switches
export SAH_PROGRAM_RATCHET_MODE=strict_single
export SAH_REQUIRE_STRICT_RATCHET=1
export SAH_TASK_TEXT_ENFORCE=1

sbatch scripts/outer_round.sbatch
