# `src/` module map

Frozen-executor bilevel RL. **Inner** = `M0 + H2 -> solution + reward` (executor
weights never change). **Outer** = `M_phi + H1 -> K candidate H2` (this is what
GRPO trains). Read `../plan.md` for the full spec.

RESTRUCTURE-001: modules are grouped by role. Every pre-restructure import
path (`outer.rewards`, `inner.session`, `python -m outer.outer_round`, ...)
still works through one-line compatibility shims that alias the moved module
object — old and new paths resolve to the SAME module.

Nothing here is dead code: modules that no other `.py` imports are loaded at
runtime through NexAU `agent.yaml` bindings or through materialized candidate
packages. Grep a binding string (e.g. `inner.harness.tools.discovery:edit_solution`)
to see who pulls a module in.

## inner/ — the executor side

| subpackage | contents | role |
|---|---|---|
| `tasks/` | `eft_task.py` | task registry: `EFTTask`, `load_tasks`, `get_task`; seed program / evaluator / spec locations for the three on-disk task families |
| `editing/` | `program_edit.py` | EVOLVE-BLOCK split/assemble, SEARCH/REPLACE diff apply, full-rewrite parse |
| `evaluation/` | `eval_runner.py`, `_eval_worker.py` | subprocess-isolated official evaluation (`EvalOutcome`, `subsample=` probes); the worker also carries the anti-reward-hacking evaluator guards (EPLB / PRISM / Txn) |
| `runtime/` | `session.py`, `harness_sdk.py`, `harness_runner.py`, `package_hash.py` | run state + budget ledger + component/tool-gate audits; the generated-tool capability surface (`ToolContext`); the NexAU executor driver (runtime component contract, transport fixes); canonical location-independent H2 hashing |
| `cli/` | `run_baseline.py` | `python -m inner.run_baseline` — M0 + one H2 over tasks, with seed/H2 provenance and trajectory retention |
| `harness/` | NexAU package | the built-in H2 (agent.yaml + tools + skills + middleware) — **frozen surface, never moved**: agent.yaml bindings and historical artifacts reference these module paths |
| `harness_candidate/` | placeholders | materialized-candidate layout documentation |

## outer/ — the proposer side

| subpackage | contents | role |
|---|---|---|
| `genome/` | `harness_spec.py` | the typed H2 genome: fail-closed schema validation, base merging, component lineage, package read-back |
| `workspace/` | `h2_workspace.py`, `propose_session.py` | H1's candidate-isolated draft filesystem (mutable-path allowlist, 8-step validation) + the inspect->edit->validate->submit session protocol |
| `proposing/` | `propose.py`, `proposer_io.py`, `forced_tool_starter.py` | per-slot H1 agent runs + the 9B-family repair pass (REPAIR-001); round-context/user-message rendering; the pre-seeded starter tool for forced slots (BAILOUT-005) |
| `compiling/` | `materialize.py` | deterministic spec -> runnable NexAU package compiler (component inventory cross-check, `_sah_py_path` injection) |
| `safety/` | `static_gates.py`, `leak_guard.py`, `tool_schema_guard.py`, `reviewer/` | AST gates for generated code; anti-leak / anti-injection text guards; H1 tool-argument repair/refusal; subprocess self-test sandbox |
| `reward/` | `rewards.py`, `program_ratchet.py`, `trajectory_budget.py` | reward/advantage implementations (legacy/v2/v3/paired/anchored, repaired-slot minimum-reward rule); the strict program incumbent ratchet; rollout-slot budgeting (fixed slots, fallback prohibition, min-valid omission) |
| `rounds/` | `outer_round.py`, `campaign_config.py`, `task_text_registry.py`, `analysis.py` | the propose/collect round CLI (proposal gate, provenance snapshots); campaign YAML -> env (fail-closed on unknown keys); task-text pinning; the frozen-M0 analysis brief |
| `harness/` | NexAU package | H1 itself — **frozen surface, never moved** |

## training/

| file | role |
|---|---|
| `grpo_to_replay.py` | `grpo_batch.jsonl` -> Weave slime replay rows: NexAU->OpenAI message conversion, GRPO group keys (`metadata.seed`), zero-advantage filtering, loss-mask close turns |

## Conventions

- The two NexAU packages (`inner/harness`, `outer/harness`) and their dotted
  module paths are frozen; everything else lives in a themed subpackage.
- New code imports the canonical subpackage path; the flat shims exist for
  scripts, tests, historical snapshots, and run artifacts.
- Anything that can manufacture reward fails closed; anything that only
  affects observability fails open (PIPELINE_FIX_LOG.md documents each rule).
