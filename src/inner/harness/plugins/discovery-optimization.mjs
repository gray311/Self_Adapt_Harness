/** Base H2 skill, implemented as an automatically enacted Cordis prompt plugin. */
export const name = 'sah-skill-discovery-optimization'
export const inject = ['systemPrompt']

export function apply(ctx) {
  ctx.systemPrompt.section({
    name: 'sah:skill:discovery-optimization',
    order: 20,
    text: `# Discovery optimization playbook

Read what the score rewards, whether the underlying quantity is maximized or
minimized, every hard validity constraint, the evaluator time limit, and the
fixed entry-function contract before editing.

Spend evaluations deliberately. Each edit must encode one concrete algorithmic
hypothesis. Prefer exact SEARCH/REPLACE blocks that preserve working code;
reserve a full-block rewrite for a genuine structural change. Treat returned
score, validity, and errors as the only evidence.

Recover deliberately: diagnose a validity failure from its error; after a valid
regression, return to the best-so-far approach and explore a different
direction. Prefer deterministic constructions that fit comfortably inside the
time limit. Consolidate when evaluations are low, and call finish when the
budget is exhausted or no credible improvement remains.`,
  })
}
