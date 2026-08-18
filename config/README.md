# Campaign config

One YAML file drives one campaign. Run:

```bash
bash scripts/run_campaign.sh config/examples/sah_v3_deep.yaml
```

The config is the single control surface. `src/outer/campaign_config.py` loads
it, validates against a schema (**unknown keys fail closed** — a typo like
`sampling.k` is rejected before a 3-hour round), fills defaults that reproduce
the current **sah v3** behavior byte-for-byte, and emits the env knobs the
pipeline already reads. A minimal config (`task` + `round_base` only) == the
current push.

## Sections

| key | default | meaning |
|---|---|---|
| `protocol` | `sah` | `sah` \| `adaptive_v1` (a bundle label; features are set individually) |
| `task`, `rounds`, `round_base` | — / 12 / — | task id, step budget, disjoint round-range start (**task + round_base required**) |
| `phi.resume_export/_ckpt` | none | continue a φ lineage after a crash instead of retraining from base |
| `sampling.K` | 8 | candidates per round (width) |
| `sampling.mode` | `parallel` | `sequential` → later samples see prior valid actions (**Feature A**) |
| `sampling.max_evals` | 30 | inner eval budget (depth) |
| `sampling.eval_timeout` | 180 | per-eval wall cap (300 for depth-limited tasks) |
| `sampling.force_tool_frac` | 0.25 | fraction of K forced to add a new tool |
| `reward.impl` | `v3` | `paired` (same-seed parent control; causal proposer credit) \| `v2` \| `v3` \| `anchored` \| `legacy` |
| `reward.hist_lambda` | 0.3 | v3 rescue weight |
| `reward.min_paired_effect` | 0.0 | minimum raw candidate-minus-parent score gain required for paired H2 promotion |
| `reward.paired_repeats` | 1 | matched decode seeds per candidate; use 3+ when estimating a stable causal effect |
| `analysis.enabled` | `false` | **Feature B**: two read-only sub-agents distill a bounded brief for the proposer |
| `sequential.enabled` | `false` | **Feature A** (auto-on when `sampling.mode=sequential`) |
| `leakage_guard.*` | on | **Feature C**: claim-word neutralization + leak-marker redaction |
| `training.plateau_rounds` | 1 | **Feature D**: 1 = train every round; N = defer until N-round confirmed plateau |
| `training.kl_coef/num_epoch` | 0.05 / 3 | GRPO knobs |

## Anti-leak invariant

Every optional feature preserves the hard rule (`[[no-strong-model-leakage]]`):
the analysis sub-agents are the **same frozen M0** the executor uses (never a
stronger model), read only our own measured telemetry, are tool-free, and every
emitted line passes `leak_guard.sanitize`. No feature can inject solution
knowledge M0 must derive itself.

## Examples

- `examples/sah_v3_deep.yaml` — the current DEEP push (all features off).
- `examples/ac2_paired.yaml` — AC2 with three matched parent-H2 seeds per
  candidate. It spends `K` proposer + `3K`
  candidate-executor + `3K` control-executor trajectories per round.
- `examples/adaptive_full.yaml` — all Adaptive-ported features on, for A/B on a
  stuck task (same task, disjoint round range, run alongside the sah baseline).
