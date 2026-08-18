# Inner loop: M0 + Cordis H2

The frozen executor model runs one native Cordis composition over an EFT task.
The baseline evolvable package is:

```text
harness/
├── cordis.yml
└── plugins/
    └── discovery-optimization.mjs
```

`cordis.yml` owns the executor persona, sampling, iteration and middleware
policy, core tool descriptions, and generated plugin mounts. Candidate H2s use
the same layout. The trusted `sah-bridge` plugin is staged separately so H1
cannot replace it.

Cordis exposes four core tools: `edit_solution`, `evaluate_solution`,
`probe_solution`, and concluding tool `finish`. The Python `InnerSession`
retains program state, best valid score, evaluation/probe budgets, component
participation, and checkpoints. Every model/tool/session event is retained in
the Cordis JSONL log and projected into SAH's replay trajectory.

Run a baseline against an OpenAI-compatible model endpoint:

```bash
PYTHONPATH=src python3 -m inner.cli.run_baseline \
  --ids eft__math__circle_packing \
  --base-url http://127.0.0.1:8800/v1 \
  --model qwen3.5-9b --max-evals 10 --require-trajectory
```

`cordis/bootstrap.sh` installs checksum-pinned Node, DSH, and Cordis versions.
No Python agent-framework dependency is needed.
