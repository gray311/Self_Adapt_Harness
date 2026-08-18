# Outer loop: M_phi + Cordis H1 -> Cordis H2

For each task, the fixed H1 gives the proposer an isolated copy of the current
H2. The proposer predicts and edits exactly:

```text
cordis.yml
plugins/*.mjs
```

The six H1 tools are read-only inspection, file write/edit/delete, validation,
and concluding submission. Validation parses the composition back into
`h2spec/2.0-cordis`, checks component/persona consistency, applies a narrow
JavaScript capability gate, recompiles canonically, and compares all live
composition/plugin bytes. Invalid, unsafe, orphan, missing, and no-op candidates
fail closed.

Generated component semantics:

- tools register with `ctx.tools.register` and may read only the narrow
  `sahBridge` snapshot;
- skills are deterministic `systemPrompt.section` plugins and are auto-enacted;
- middleware uses the Cordis `agent/pre-step` waterfall and preserves `next()`;
- the trusted bridge, evaluator budget, and model endpoint are not proposer
  writable.

Each accepted H2 is evaluated by frozen M0. Paired controls use the same task,
program, model, budget, and decode seed. GRPO trains only the proposer adapter;
M0 stays frozen.

```bash
PYTHONPATH=src python3 -m outer.rounds.outer_round --help
```

Round outputs keep proposer Cordis session paths, candidate package hashes,
component lineage, H2 trajectories, paired rewards, and replay rows.
