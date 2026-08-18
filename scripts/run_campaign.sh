#!/usr/bin/env bash
# One command, one YAML config, one campaign.
#   run_campaign.sh <config.yaml>
# The YAML is the single control surface (see src/outer/rounds/campaign_config.py and
# config/examples/). This script loads it, exports the env knobs the pipeline
# already reads (K, MAX_EVALS, EVAL_TIMEOUT, SAH_ADV, SAH_SEQUENTIAL,
# SAH_ANALYSIS, PLATEAU_ROUNDS, RESUME_PHI, ...), and drives fresh_campaign.
# Defaults reproduce sah v3, so a minimal config == the current push.
set -uo pipefail
source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
SAH="${SAH_REPO:-$CODE_ROOT/sah_corids}"
[ -d "$SAH" ] || SAH="$CODE_ROOT/self_adapt_harness"
CFG="${1:?usage: run_campaign.sh <config.yaml>}"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 1; }

# validate + translate the YAML into env (fails closed on typos/missing keys)
ENV_LINES=$(cd "$SAH/src" && python3 -m outer.rounds.campaign_config "$CFG") || {
  echo "[run_campaign] config invalid:" >&2; echo "$ENV_LINES" >&2; exit 1; }

# read task/rounds/round_base/workspace directly (not env-mapped)
read TASK ROUNDS RBASE WS < <(cd "$SAH/src" && python3 - "$CFG" <<'PY'
import sys; sys.path.insert(0, ".")
from outer import campaign_config as cc
c = cc.load(sys.argv[1]).data
ws = c.get("workspace") or ""
print(c["task"], c["rounds"], c["round_base"], ws)
PY
)
[ -n "$WS" ] || WS="$RUN_ROOT/self_adapt_harness/fresh_all/$TASK"
mkdir -p "$WS"

# export every knob the config emits
while IFS='=' read -r k v; do [ -n "$k" ] && export "$k=$v"; done <<< "$ENV_LINES"

echo "[run_campaign] $CFG"
echo "[run_campaign] task=$TASK rounds=$ROUNDS round_base=$RBASE ws=$WS"
echo "[run_campaign] K=$K max_evals=$MAX_EVALS eval_timeout=$EVAL_TIMEOUT adv=$SAH_ADV"
echo "[run_campaign] features: seq=$SAH_SEQUENTIAL analysis=$SAH_ANALYSIS leak_neutralize=$SAH_LEAK_NEUTRALIZE plateau=${PLATEAU_ROUNDS:-1}"

# warm-start bases from the best evolved package for this task (shared with sota_push)
python3 - "$TASK" "$WS" <<'PY'
import json, glob, os, sys
task, ws = sys.argv[1], sys.argv[2]
OUT = os.path.expandvars("$RUN_ROOT/self_adapt_harness/outer")
best = None
for nb in glob.glob(OUT + "/round*/next_bases.json"):
    try: e = json.load(open(nb)).get(task)
    except Exception: continue
    if e and os.path.isdir(e.get("package", "")) and \
       (best is None or e.get("score", -9e9) > best.get("score", -9e9)):
        best = e
f = os.path.join(ws, "round000_bases.json")
if best:
    if os.path.exists(f) and not os.path.exists(f + ".orig"): os.rename(f, f + ".orig")
    json.dump({task: best}, open(f, "w"), indent=1)
    print("[run_campaign] warm-start base score=%s" % best.get("score"))
PY

# seed best_programs.json for a fresh workspace from the global inheritance so
# rollouts start from the current best program (fair A/B) and don't crash on a
# missing seed file. Only creates it if absent (never clobbers an evolving one).
python3 - "$TASK" "$WS" <<'PY'
import json, os, sys
task, ws = sys.argv[1], sys.argv[2]
dst = os.path.join(ws, "best_programs.json")
if not os.path.exists(dst):
    src = os.path.expandvars("$RUN_ROOT/self_adapt_harness/outer/best_programs.json")
    ent = {}
    try:
        g = json.load(open(src)).get(task)
        if g: ent = {task: g}
    except Exception:
        pass
    json.dump(ent, open(dst, "w"), indent=1)
    print("[run_campaign] seeded best_programs.json (%s)"
          % ("from global best" if ent else "empty — will start from initial program"))
PY

rm -f "$WS/STOP" "$WS/RUNNING"; touch "$WS/RUNNING"
FTF="${FORCE_TOOL_FRAC:-0.25}"
setsid bash -c '
  source /lustre/fsw/portfolios/av/users/yingzim/config/workspace_env.sh
  bash "'"$SAH"'/scripts/fresh_campaign.sh" "'"$TASK"'" "'"$ROUNDS"'" "'"$RBASE"'" "'"$FTF"'" "'"$WS"'" \
    > "'"$WS"'/driver.log" 2>&1
  rm -f "'"$WS"'/RUNNING"
' </dev/null >/dev/null 2>&1 &
sleep 3
echo "[run_campaign] launched (driver PID group detached); log: $WS/driver.log"
