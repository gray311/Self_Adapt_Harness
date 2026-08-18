#!/usr/bin/env bash
# SOTA-push relaunch: continue a task's fresh_all campaign with the historical-
# frontier advantage (SAH_ADV=v3) + wide sampling (K=16) + deeper inner search
# (MAX_EVALS=30). The program ratchet (best_programs.json in the task workspace)
# carries the current-best solution forward; phi restarts from base to avoid the
# trained phi getting anchored on a plateau. Detached via setsid.
#   sota_push.sh <task_id> <round_base> [nsteps=12]
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="$CODE_ROOT/self_adapt_harness"
FRESH="$RUN_ROOT/self_adapt_harness/fresh_all"
TASK="$1"; RBASE="$2"; NST="${3:-12}"
d="$FRESH/$TASK"

# guard: don't double-launch
if [ -f "$d/RUNNING" ] && ps -eo cmd | grep -q "[f]resh_campaign.sh $TASK"; then
  echo "[sota_push] $TASK already running — abort"; exit 1
fi
# AHC needs native aarch64 eval env
if [[ "$TASK" == *ahc* ]]; then
  export AHC_NATIVE=1 AHC_CXX=g++ AHC_CASE_WORKERS=12 AHC_CACHE_DIR="$SAH/ahc_work/cache"
fi
# archive prior driver log, clear stale flags
n=$(ls "$d"/driver.push*.log 2>/dev/null | wc -l)
[ -f "$d/driver.log" ] && mv "$d/driver.log" "$d/driver.push$((n+1)).log"
rm -f "$d/STOP" "$d/RUNNING"; touch "$d/RUNNING"

# warm-start the step-1 bases: WITHOUT this, fresh_campaign step 1 reads the
# workspace's original round000_bases.json, whose package pointer is the
# INITIAL harness and whose score is the initial baseline — i.e. a relaunch
# silently discards the evolved-harness ratchet AND deflates the step-1 base
# (observed: erdos push step1 base=0.834 vs ratcheted 0.9996, package reset to
# src/inner/harness). Point it at the max-score next_bases entry instead.
python3 - "$TASK" "$d" <<'PYEOF'
import json, glob, sys, os
task, ws = sys.argv[1], sys.argv[2]
OUT = os.path.expandvars("$RUN_ROOT/self_adapt_harness/outer")
best = None
for nb in glob.glob(OUT + "/round*/next_bases.json"):
    try:
        e = json.load(open(nb)).get(task)
    except Exception:
        continue
    if e and os.path.isdir(e.get("package", "")) and \
       (best is None or e.get("score", -9e9) > best.get("score", -9e9)):
        best = e
f = os.path.join(ws, "round000_bases.json")
if best:
    if os.path.exists(f) and not os.path.exists(f + ".orig"):
        os.rename(f, f + ".orig")   # keep the original for provenance
    json.dump({task: best}, open(f, "w"), indent=1)
    print("[sota_push] warm-start base: score=%s\n[sota_push]   package=%s"
          % (best.get("score"), best.get("package")))
else:
    print("[sota_push] no prior next_bases found — keeping existing bases file")
PYEOF

# width-vs-depth knobs (measured 2026-07-29: erdos/AC1/AC2 winners' eval walls
# sit at 84-95% of the timeout cap -> DEPTH-limited; trade K for EVAL_TIMEOUT
# there. ahc039 is eval-noise-dominated -> keep WIDE. hadamard is
# exploration-limited -> neither helps, needs elite diversity.)
PUSH_K="${PUSH_K:-16}"; PUSH_EVALS="${PUSH_EVALS:-30}"; PUSH_TIMEOUT="${PUSH_TIMEOUT:-180}"
setsid bash -c '
  source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
  d="'"$d"'"
  SAH_ADV=v3 SAH_HIST_LAMBDA=0.3 K='"$PUSH_K"' MAX_EVALS='"$PUSH_EVALS"' EVAL_TIMEOUT='"$PUSH_TIMEOUT"' \
    bash "'"$SAH"'/scripts/fresh_campaign.sh" "'"$TASK"'" "'"$NST"'" "'"$RBASE"'" 0.25 "$d" \
    > "$d/driver.log" 2>&1
  rm -f "$d/RUNNING"
' </dev/null >/dev/null 2>&1 &
sleep 3
echo "[sota_push] launched $TASK: SAH_ADV=v3 K=$PUSH_K MAX_EVALS=$PUSH_EVALS EVAL_TIMEOUT=$PUSH_TIMEOUT, $NST steps @ round$RBASE+"
echo "[sota_push] RUNNING now: $(ls "$FRESH"/*/RUNNING 2>/dev/null | wc -l) = $(( $(ls "$FRESH"/*/RUNNING 2>/dev/null | wc -l) * 4 )) GPU"
