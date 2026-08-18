# SAH Cordis runtime

This directory owns SAH's pinned Cordis runtime. It runs the official DeepSeek
Harness (`dsh`) headless profile, whose agent runtime is a Cordis plugin tree,
and routes model calls to SAH's OpenAI-compatible vLLM endpoint.

Pinned runtime:

- `@deepseek-ai/dsh@0.1.0-rc.7`
- `@deepseek-ai/cordis@4.0.1`
- Node.js `22.19.0`, checksum-verified on x86_64 and aarch64

## Run

From the repository root, first run the deterministic end-to-end smoke test:

```bash
./cordis/smoke.sh
```

It starts a local OpenAI-compatible mock model and verifies the complete path:
Cordis boot, plugin dependency settlement, DSH agent loop, model request,
stream assembly, durable session events, and final assistant text.

Then run the SAH-specific H2 regression, which additionally exercises the
authenticated Python tool bridge and a concluding `finish` call:

```bash
PYTHONPATH=src python3 tests/run_cordis_h2_smoke.py
```

Against SAH's normal vLLM service:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8800/v1 \
SAH_CORDIS_MODEL=qwen3.5-9b \
OPENAI_API_KEY=EMPTY \
./cordis/real_smoke.sh
```

Or allocate one GPU, start Qwen3.5-9B, and run the same check in one bounded
Slurm job:

```bash
sbatch cordis/slurm/real_model_smoke.sbatch
```

Its vLLM log, isolated DSH state, and `result.txt` are written under
`.runtime/cordis/real-model-$SLURM_JOB_ID/`. The same directory contains a
readable, one-event-per-line `trajectory.jsonl` and a validated
`trajectory.manifest.json` with its hash, model route, event counts, and
completion status.

Run an arbitrary one-shot headless task with:

```bash
./cordis/run.sh "summarize this workspace"
```

Inspect the fully composed plugin tree without making a model call:

```bash
./cordis/run.sh --dump-config
```

## Configuration

| variable | default | purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8800/v1` | OpenAI-compatible endpoint |
| `OPENAI_API_KEY` | `EMPTY` | credential reference; never written to YAML |
| `SAH_CORDIS_MODEL` | `qwen3.5-9b` | exact served model id |
| `SAH_CORDIS_CONTEXT_WINDOW` | `131072` | model capacity advertised to Cordis |
| `SAH_CORDIS_MAX_TOKENS` | `8192` | per-request output cap |
| `SAH_CORDIS_MAX_RETRIES` | `2` | provider-request retries; never replays a Cordis turn |
| `DSH_HOME` | `.runtime/cordis/dsh-home` | profiles, sessions, settings, credentials |
| `SAH_CORDIS_TRAJECTORY_ROOT` | `$DSH_HOME/trajectories` | raw lossless session JSONL root |
| `SAH_CORDIS_RUNTIME_DIR` | `.runtime/cordis` | architecture-specific Node/npm runtime |

`bootstrap.sh` installs separate `linux-x64` and `linux-arm64` closures under
the runtime directory. This matters because the login node and GB200 workers
use different architectures. Runtime files are ignored by Git.

## SAH boundary

Both live agent roles use this runtime:

- H2 registers `edit_solution`, `evaluate_solution`, `probe_solution`, and
  concluding `finish`; evaluator and budget state remain in `InnerSession`.
- H1 registers the six inspect/edit/validate/submit tools over a private native
  Cordis workspace. Its mutable output is only `cordis.yml` and
  `plugins/*.mjs`.

`plugins/sah-bridge.mjs` is trusted and snapshotted separately from candidate
plugins. It uses a rollout-local bearer token and loopback server, records tool
lifecycle participation, applies iteration/budget middleware, and never exposes
credentials or Python callables to the proposer. Raw Cordis JSONL is the
authoritative trajectory; `src/cordis_runtime/trajectory.py` supplies the
stable message projection consumed by existing reward and replay code.
