# SAH — Self-Adapt Harness on Cordis

A permanently frozen executor solves discovery tasks inside an **agent
harness**; a trained proposer rewrites that harness. The executor's weights
(`M0`, Qwen3.5-9B) never change, so every gain is attributable to the harness
itself.

```text
inner loop:   M0    + H2       ->  solution + reward
outer loop:   M_phi + H1(tau)  ->  K candidate H2 packages
```

For one task instance `tau`, the fixed proposer harness **H1** produces `K = 8`
task-conditioned **H2** candidates. Each candidate drives one frozen-`M0`
rollout. The rewards form one instance-wise GRPO group that trains only
`M_phi`.

## Cordis harness boundary

H1 and H2 both run through the official DeepSeek Harness headless runtime,
whose agent runtime is a Cordis plugin tree. Cordis owns the model loop and
the plugin lifecycle; the evaluator, the edit ledger, the workspace protocol
and all reward bookkeeping stay in Python.

Each rollout therefore gets a short-lived HTTP bridge bound to `127.0.0.1`
and protected by a random bearer token (`src/cordis_runtime/bridge.py`).
Calls are serialised because `InnerSession` and `ProposeSession` are
deliberately single-agent state machines. What the proposer may change is
declared by the `h2spec/2.0-cordis` genome and materialised into a runnable
Cordis composition — the proposer edits *files*, never the runtime.

## What a proposal may change

| dimension | what the proposer writes |
|---|---|
| prompt | the executor's system prompt |
| skills | a skill plugin the runtime enacts into context |
| tools | a generated tool: implementation, schema, and its mount |
| middleware | a `before_model` hook — the only channel that costs the executor no turn |
| parameters | sampling and budget settings, within genome-validated ranges |

## Layout

| path | role |
|---|---|
| `cordis/` | checksum-pinned Node/DSH/Cordis runtime, the trusted bridge plugin, smoke tests, and Slurm launchers |
| `src/cordis_runtime/` | isolated launcher, authenticated Python bridge, OpenAI request adapter, trajectory projection |
| `src/inner/` | task and evaluator pipeline, plus the baseline H2 `src/inner/harness/cordis.yml` and its skill plugin |
| `src/outer/` | the fixed H1 composition, the `h2spec/2.0-cordis` genome, workspace validator and materializer, rewards, and round orchestration |
| `src/training/` | Cordis-trajectory to Weave replay conversion and the proposer LoRA driver |
| `scripts/` | cluster launch, runtime, provenance, and analysis helpers |
| `config/` | campaign knobs (`config/campaign.yaml`) and the YAML-to-env loader |

## Quick verification

The deterministic checks need no GPU:

```bash
./cordis/smoke.sh
```

Against an existing OpenAI-compatible vLLM service:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8800/v1 \
SAH_CORDIS_MODEL=qwen3.5-9b \
./cordis/real_smoke.sh
```

## Running a round

```bash
export PYTHONPATH=src
export SAH_DATASET_ROOT=/path/to/prepared/dataset

# 1. K candidate harnesses for one task
python3 -m outer.rounds.outer_round propose \
  --round-dir runs/round000 --tasks <task_id> --k 8 \
  --base-url http://127.0.0.1:8800/v1 --model <served-model>

# 2. roll out each candidate, plus a same-seed rollout of its parent harness
python3 -m inner.cli.run_baseline \
  --harness-dir runs/round000/tasks/<task_id>/cand00 --ids <task_id> \
  --base-url http://127.0.0.1:8800/v1 --model <served-model> \
  --max-evals 20 --seed 420000 --out runs/round000/rollouts/<task_id>/cand00

# 3. paired rewards, ratchet, next round's bases
python3 -m outer.rounds.outer_round collect --round-dir runs/round000
```

A task with no baseline entry cold-starts: round 0 begins at 0.0 and the
paired controls of that round replace the placeholder before round 1 inherits
it, so a newly onboarded task needs no pre-registered score.

## What the framework guarantees

* **Causal attribution.** Every candidate is scored against a rollout of its
  own parent harness at the **same decode seed**, and the pair must satisfy a
  provenance contract (same model, same budget, stable package hashes) before
  any credit is assigned. If the executor would have found the same program
  anyway, the harness earns nothing.
* **Fail-closed accounting.** A round that cannot produce enough valid
  proposals fails rather than quietly rolling out the incumbent; invalid slots
  are charged and never substituted.
* **Anti-reward-hacking.** An evaluation the harness marks invalid can never
  become the best score, however high it reads; a package that mutates itself
  mid-rollout is score-ineligible; generated tool code passes a static gate
  before it can run.
* **No strong-model leakage.** Repair of a failed proposal is performed by the
  frozen executor model itself, and a repaired slot trains the proposer at the
  minimum reward — it never earns credit for a submission it did not write.

## External dependencies

- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) and
  [Cordis](https://github.com/cordiverse/cordis), installed from the committed
  lockfile by `cordis/bootstrap.sh`;
- `Weave_v2` for offline GRPO/LoRA training;
- [evolution-fine-tuning](https://github.com/Open-Galapagos/evolution-fine-tuning)
  evaluator data under `$DATASET_ROOT/self_adapt_harness/`;
- Qwen3.5-9B served through vLLM with Qwen XML tool-call parsing.

Cluster locations come from `$CODE_ROOT`, `$DATASET_ROOT`, `$MODEL_ROOT` and
`$RUN_ROOT`. Runtime downloads, DSH profiles, sessions, checkpoints, datasets
and full trajectories stay out of Git.

## Documentation

- [`cordis/README.md`](cordis/README.md) — the pinned runtime.
- [`src/README.md`](src/README.md) — module and trust-boundary map.
- [`src/inner/README.md`](src/inner/README.md) — model and H2 execution.
- [`src/outer/README.md`](src/outer/README.md) — proposer to Cordis package flow.
- [`config/README.md`](config/README.md) — campaign knobs.
