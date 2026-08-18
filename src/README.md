# `src/` module map

Frozen-executor bilevel RL. **Inner** = `M0 + H2 -> solution + reward`;
**outer** = `M_phi + H1 -> K candidate H2`.

Both H1 and H2 now run through DeepSeek Harness/Cordis. Their mutable harness
surface is native and deliberately small:

```text
cordis.yml
plugins/*.mjs
```

The model endpoint, evaluator state, budgets, and Python functions stay behind
the trusted loopback bridge in `cordis/plugins/sah-bridge.mjs` and
`cordis_runtime/`.

## Inner executor

| package | role |
|---|---|
| `inner/tasks/` | EFT task registry and seed/evaluator locations |
| `inner/editing/` | EVOLVE-BLOCK and SEARCH/REPLACE editing |
| `inner/evaluation/` | subprocess-isolated official evaluation |
| `inner/runtime/` | session ledger, H2 Cordis runner, package hash |
| `inner/harness/` | baseline H2 `cordis.yml` and base skill plugin |
| `inner/cli/` | baseline CLI and artifact/provenance writing |

## Outer proposer

| package | role |
|---|---|
| `outer/genome/` | `h2spec/2.0-cordis`, inheritance, lineage, read-back |
| `outer/workspace/` | isolated Cordis filesystem and inspect/edit/validate/submit protocol |
| `outer/proposing/` | H1 Cordis runs, component scaffolds, proposer messages |
| `outer/compiling/` | deterministic genome -> Cordis package materializer |
| `outer/safety/` | native plugin capability gates and leak guards |
| `outer/reward/` | paired rewards, program ratchet, trajectory accounting |
| `outer/rounds/` | round orchestration and campaign analysis |
| `outer/harness/` | fixed H1 Cordis composition and system instructions |

## Shared Cordis boundary

`cordis_runtime/runner.py` snapshots plugins into an isolated DSH profile,
starts authenticated Python and model proxies, invokes the pinned Cordis CLI,
and projects the durable Cordis JSONL session into the existing SAH trajectory
shape. `training/grpo_to_replay.py` consumes that shape with the same six H1
tool schemas used at inference.

Flat compatibility modules remain for historical scripts, but no live H1/H2
execution path imports the former agent framework.
