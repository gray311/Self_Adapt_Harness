# SAH — Self-Adapt Harness

A permanently frozen executor solves discovery tasks inside an **agent
harness**; a trained proposer rewrites that harness. The executor's weights
(`M0`, Qwen3.5-9B) never change, so every gain is attributable to the harness
itself.

```
        proposes a new harness                     runs the task
M_phi ─────────────────────────────▶  H2 package  ──────────────▶  score
(trained proposer, H1)              (prompt, tools,   (frozen executor M0)
        ▲                            skills, middleware,
        │  paired causal reward       sampling, budgets)
        └──────────────────────────────────────────────────────────┘
             candidate vs SAME-SEED rollout of its parent harness
```

Two loops:

* **inner** — the executor drives an `edit_solution → evaluate_solution` loop
  over one task under a fixed evaluation budget, inside the H2 package.
* **outer** — the proposer edits a private copy of H2 as *files*
  — the package's own `prompt.md`, `tools/`, `custom_tools/`, `skills/`,
  `middlewares/` and `agent.yaml` — validates it, and submits. For one task instance the fixed
  proposer harness H1 produces `K` candidates whose rewards form one
  instance-wise GRPO group that trains only `M_phi`.

## What a proposal may change

| dimension | what the proposer writes |
|---|---|
| prompt | the executor's system prompt |
| skill | `skills/<slug>/SKILL.md`, auto-enacted into the executor's context |
| tool | `custom_tools/<name>.py` plus its schema and mount |
| middleware | a `before_model` hook — the only capability channel that costs the executor no turn |
| parameters | `sampling.*`, `agent.max_iterations` and built-in middleware policy, within genome-validated ranges |

The proposer works through a constrained tool surface — `harness_shell`,
`write_harness_file`, `edit_harness_file`, `delete_harness_file`,
`validate_harness`, `submit_harness` — so a proposal is a set of file edits
that must pass the genome validator before it can run.

## Layout

| path | role |
|---|---|
| `src/inner/` | H2: the fixed executor harness, its runtime and session, tasks, evaluation |
| `src/outer/genome/` | the harness genome — what a proposal may change, and its caps |
| `src/outer/proposing/` | H1: the proposer agent, its briefing, and its per-slot exploration roles |
| `src/outer/workspace/` | the file-native editing session the proposer works in |
| `src/outer/compiling/` | genome → runnable H2 package on disk |
| `src/outer/reward/` | paired causal rewards, program ratchet, trajectory budget |
| `src/outer/rounds/` | one evolve round: propose → rollout → collect |
| `src/outer/safety/` | static gates, tool-schema guard, leak guard |
| `src/training/` | GRPO batch export for offline proposer training |
| `scripts/` | cluster launch, runtime, provenance, and analysis helpers |
| `config/` | campaign knobs (`config/campaign.yaml`) and the YAML-to-env loader |

## Running a round

```bash
export PYTHONPATH=src
export SAH_DATASET_ROOT=/path/to/prepared/dataset

# 1. K candidate harnesses for one task
python3 -m outer.rounds.outer_round propose \
  --round-dir runs/round000 --tasks <task_id> --k 8 \
  --base-url http://127.0.0.1:8000/v1 --model <served-model>

# 2. roll out each candidate, plus a same-seed rollout of its parent harness
python3 -m inner.cli.run_baseline \
  --harness-dir runs/round000/tasks/<task_id>/cand00 --ids <task_id> \
  --base-url http://127.0.0.1:8000/v1 --model <served-model> \
  --max-evals 20 --seed 420000 --out runs/round000/rollouts/<task_id>/cand00

# 3. paired rewards, ratchet, next round's bases
python3 -m outer.rounds.outer_round collect --round-dir runs/round000
```

A task with no baseline entry cold-starts: round 0 begins at 0.0 and the
paired controls of that round replace the placeholder before round 1 inherits
it, so a newly onboarded task needs no pre-registered score. Point
`SAH_BASELINE_JSON` at a baseline file to start from measured scores instead.

## Configuration

`config/campaign.yaml` is the readable view of every knob, each marked
**tunable** (an environment variable) or **fixed** (a code constant, with the
file named). `source config/load_env.sh` exports the tunable ones.

| knob | effect |
|---|---|
| `SAH_PROPOSAL_GATE` | how many of the K proposals must be valid before a round may proceed |
| `SAH_PAIRED_REPEATS` | roll out each candidate/control pair R times; the causal delta is the mean |
| `SAH_H1_DIVERSITY` | per-slot exploration roles (tool / skill / middleware / params / rewrite) |
| `SAH_NOVELTY_GUARD` | refuses re-proposing a component already measured causally futile |
| `CURVE_INHERIT_BEAM` | keep a diverse runner-up lineage alongside the promoted one |
| `CURVE_COMPONENT_RATCHET` | inherit a component whose isolated causal effect clears the threshold |

## What the framework guarantees

* **Causal attribution.** Every candidate is scored against a rollout of its
  own parent harness at the **same decode seed**, and the pair must satisfy a
  provenance contract (same model, same budget, stable package hashes) before
  any credit is assigned. If the executor would have found the same program
  anyway, the harness earns nothing.
* **Selection at two levels.** A candidate harness is promoted only when it
  beats both the incumbent score and its own control; separately, a component
  whose *isolated* effect clears the threshold is grafted into the next
  round's base on its own merit, with every admission and refusal recorded
  with its evidence.
* **Fail-closed accounting.** A round that cannot produce enough valid
  proposals fails rather than quietly rolling out the incumbent; invalid slots
  are charged and never substituted.
* **Anti-reward-hacking.** An evaluation the harness marks invalid can never
  become the best score, however high it reads; a package that mutates itself
  mid-rollout is score-ineligible; generated tool code passes a static gate
  before it can run; a middleware that never fired makes its rollout
  ineligible rather than silently earning credit.
* **No strong-model leakage.** Repair of a failed proposal is performed by the
  frozen executor model itself, and a repaired slot trains the proposer at the
  minimum reward — it never earns credit for a submission it did not write.

## External dependencies

- the NexAU agent framework, which hosts both harnesses;
- `Weave_v2` for offline GRPO/LoRA training;
- [evolution-fine-tuning](https://github.com/Open-Galapagos/evolution-fine-tuning)
  evaluator data under `$DATASET_ROOT/self_adapt_harness/`;
- Qwen3.5-9B served through vLLM with Qwen XML tool-call parsing.

Cluster locations come from `$CODE_ROOT`, `$DATASET_ROOT`, `$MODEL_ROOT` and
`$RUN_ROOT`. Sessions, checkpoints, datasets and full trajectories stay out of
Git.

## Documentation

- [`src/README.md`](src/README.md) — module and trust-boundary map.
- [`src/inner/README.md`](src/inner/README.md) — model and H2 execution.
- [`src/outer/README.md`](src/outer/README.md) — proposer to H2 package flow.
- [`src/training/README.md`](src/training/README.md) — GRPO replay export.
- [`config/README.md`](config/README.md) — campaign knobs.
