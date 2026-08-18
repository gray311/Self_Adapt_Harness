#!/usr/bin/env python3
r"""Build a TTT-Discover-style self-training set for the EXECUTOR.

This is the "update the executor" arm of the score-compute comparison. It is the
same idea as test-time RL for discovery: the model is trained on its OWN
high-scoring solutions for the task, with no external solution, no stronger
model, and no human hint -- only rollouts the frozen executor already produced.

Rows are written in the trainer's replay format
    {"messages": [...], "tools": [], "metadata": {"advantage": ..., ...}}
so scripts/train_mphi_step.sh can consume them unchanged, except that the
policy being updated is the executor rather than the proposer.

Advantage is leave-one-out within a task, matching the objective used elsewhere,
and the rows are emitted in chronological order so a prefix of the file
corresponds to a prefix of the executor-rollout budget -- that is what makes the
score-vs-compute curve well defined.

  build_ttt_selftrain.py <task_id> <out.jsonl> [--max-rows N]
"""
import json, glob, os, re, sys

RUN = os.environ.get("RUN_ROOT") or "/lustre/fsw/portfolios/av/users/yingzim/runs"
SAH = os.environ.get("CODE_ROOT", "/lustre/fsw/portfolios/av/users/yingzim/code") + "/self_adapt_harness"
R = f"{RUN}/self_adapt_harness"

TASK_BLURB = {
 "eft__math__erdos_min_overlap":
   "Erdos minimum-overlap problem: choose a construction minimising the maximum overlap functional.",
 "eft__math__circle_packing":
   "Circle packing, n=26: place 26 non-overlapping circles in the unit square maximising the sum of radii.",
 "eft__math__hadamard_maximal_det":
   "Maximal-determinant Hadamard-type matrix: maximise |det| for the given order.",
 "eft__math__first_autocorr_ineq":
   "First autocorrelation inequality: minimise the achievable constant.",
 "eft__math__second_autocorr_ineq":
   "Second autocorrelation inequality: maximise the achievable constant.",
 "eft__ahc_simpletes__ahc039":
   "AtCoder Heuristic Contest 039: maximise the official 150-case score.",
}


TOOLS = [{
  "type": "function",
  "function": {
    "name": "edit_solution",
    "description": "Change the code inside the # EVOLVE-BLOCK region, then call evaluate_solution to score it.",
    "parameters": {"type": "object",
      "properties": {"code": {"type": "string",
        "description": "SEARCH/REPLACE diff block(s), or the full replacement body for the EVOLVE-BLOCK region."}},
      "required": ["code"]}}},
 {"type": "function",
  "function": {
    "name": "evaluate_solution",
    "description": "Score the current program against the task evaluator. Consumes one unit of the evaluation budget.",
    "parameters": {"type": "object", "properties": {}, "required": []}}}]


def collect(task):
    """chronological (score, program) for this task, from our own rollouts only."""
    rows = []
    for f in glob.glob(f"{R}/outer/round*/rollouts/{task}/*/*/summary.json"):
        m = re.search(r"round(\d+)", f)
        try:
            e = json.load(open(f))
        except Exception:
            continue
        e = e[0] if isinstance(e, list) else e
        p, s = e.get("best_program"), e.get("best_score")
        if not p or s is None or s <= 0:
            continue
        rows.append((int(m.group(1)) if m else 0, float(s), p))
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    task, out = sys.argv[1], sys.argv[2]
    cap = int(sys.argv[sys.argv.index("--max-rows") + 1]) if "--max-rows" in sys.argv else 10**9
    system = open(f"{SAH}/src/inner/harness/system.md").read()
    rows = collect(task)[:cap]
    if not rows:
        sys.exit(f"no self-generated rollouts for {task}")

    scores = [s for _, s, _ in rows]
    n = len(scores)
    total = sum(scores)
    written = 0
    with open(out, "w") as fh:
        for rnd, s, prog in rows:
            # leave-one-out baseline over this task's own rollouts
            loo = (total - s) / (n - 1) if n > 1 else s
            adv = s - loo
            # Qwen3.5 inline tool-call form -- the loss-mask generator only
            # finds trainable assistant tokens in this shape, and it is also what
            # the executor actually emits (edit_solution with the new block).
            call = ("<tool_call>\n<function=edit_solution>\n<parameter=code>\n"
                    f"{prog}\n</parameter>\n</function>\n</tool_call>")
            rec = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content":
                        f"Task: {TASK_BLURB.get(task, task)}\n\n"
                        "Improve the EVOLVE-BLOCK. Call edit_solution with the "
                        "new block, then evaluate_solution."},
                    {"role": "assistant", "content": call},
                    {"role": "tool", "content":
                        f"Edit applied. evaluate_solution -> combined_score {s:.6f}."},
                ],
                "tools": TOOLS,
                "metadata": {"advantage": adv, "reward": s, "task_id": task,
                             "round": rnd, "valid": True, "arm": "ttt_executor",
                             "tools": TOOLS},
            }
            fh.write(json.dumps(rec) + "\n")
            written += 1
    pos = sum(1 for _, s, _ in rows if s > (total - s) / max(n - 1, 1))
    print(f"{task}: wrote {written} rows -> {out}")
    print(f"  score range {min(scores):.6f} .. {max(scores):.6f}, {pos} above the LOO baseline")


if __name__ == "__main__":
    main()
