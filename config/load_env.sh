# Turn config/campaign.yaml's TUNABLE knobs into environment exports.
#   usage:  source config/load_env.sh   (then submit init / start the driver)
# Only env-tunable keys are exported; 'fixed' entries in the YAML are
# documentation of code constants and are ignored here.
_SAH_CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eval "$(python3 - "$_SAH_CFG_DIR/campaign.yaml" <<'PY'
import re, sys

MAP = {  # yaml path -> env var
    ("proposer_h1", "max_iters"): "SAH_H1_MAX_ITERS",
    ("proposer_h1", "no_tool_nudges"): "SAH_H1_NO_TOOL_NUDGES",
    ("proposer_h1", "diversity_roles"): "CURVE_H1_DIVERSITY",
    ("proposer_h1", "novelty_guard"): "CURVE_NOVELTY_GUARD",
    ("proposer_h1", "sequential_sampling"): "SAH_SEQUENTIAL",
    ("repair", "max_iters"): "SAH_REPAIR_MAX_ITERS",
    ("gates", "proposal_gate"): "SAH_PROPOSAL_GATE",
    ("gates", "phi_probe"): "CURVE_PHI_PROBE",
    ("paired_causal_controls", "repeats"): "CURVE_PAIRED_REPEATS",
    ("training", "lr"): "CURVE_TRAIN_LR",
    ("training", "token_budget"): "CURVE_TRAIN_TOKEN_BUDGET",
    ("driver", "round8"): "DRIVER_ROUND8",
    ("driver", "max_resubmits"): "DRIVER_MAX_RESUBMITS",
}

section = None
values = {}
for raw in open(sys.argv[1]):
    line = raw.split("#", 1)[0].rstrip()
    if not line.strip():
        continue
    m = re.match(r"^([a-z_0-9]+):\s*$", line)
    if m:
        section = m.group(1)
        continue
    m = re.match(r"^\s+([a-z_0-9]+):\s*(.+?)\s*$", line)
    if m and section:
        values[(section, m.group(1))] = m.group(2)

for key, env in MAP.items():
    if key not in values:
        continue
    v = values[key]
    if v in ("true", "false"):
        v = "1" if v == "true" else "0"
    print(f"export {env}={v}")
PY
)"
echo "[config] campaign.yaml applied:" \
  "gate=$SAH_PROPOSAL_GATE repeats=$CURVE_PAIRED_REPEATS" \
  "diversity=$CURVE_H1_DIVERSITY novelty=$CURVE_NOVELTY_GUARD" \
  "probe=$CURVE_PHI_PROBE lr=$CURVE_TRAIN_LR round8=$DRIVER_ROUND8"
