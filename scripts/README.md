# Script layout

The top level contains shell and Slurm entry points for older campaign flows.
Python utilities are grouped by responsibility:

- `runtime/`: execution-critical helpers that may be copied into immutable run
  snapshots.
- `analysis/audits/`: fail-closed result and protocol audits.
- `analysis/collect/`: artifact and scheduler-data collectors.
- `analysis/figures/`: plotting and paper-figure entry points.
- `analysis/inspect/`: interactive status and trajectory inspection.
- `analysis/reward_route/`: reward-route-specific post-processing.

New self-contained studies belong under `experiments/` rather than adding more
files here. Python filenames use `snake_case`; Slurm launchers use the same stem
as their worker, with `_worker.sh` reserved for container-side implementations.

Run Python entry points from the repository root, for example:

```bash
python3 scripts/analysis/inspect/trajectory.py <round-dir>
python3 scripts/analysis/figures/score_compute_curves.py
python3 scripts/runtime/audit_trajectories.py <rollout-dir>
```

The directories are explicit Python packages, so tests and reusable helpers use
imports such as `scripts.runtime.provenance` instead of filename-based loading.
