/**
 * Trusted SAH runtime plugin.
 *
 * DSH/Cordis owns the agent loop.  This plugin registers the fixed H1/H2 tool
 * surfaces and forwards calls to the rollout-local Python state machine over
 * an authenticated loopback endpoint.  Candidate plugins may inject the
 * narrow `sahBridge` service; they never receive the bearer token itself.
 */

export const name = 'sah-bridge'
export const inject = ['tools', 'systemPrompt']

const OBJECT = Object.freeze({ type: 'object', properties: {}, additionalProperties: false })

const INNER_TOOLS = Object.freeze([
  {
    name: 'edit_solution',
    description: 'Change the code inside the EVOLVE-BLOCK. Prefer exact SEARCH/REPLACE blocks for targeted edits; a complete replacement body is also accepted. Evaluate after editing.',
    parameters: {
      type: 'object',
      properties: {
        code: { type: 'string', description: 'SEARCH/REPLACE block(s) or the complete replacement body.' },
      },
      required: ['code'],
      additionalProperties: false,
    },
  },
  {
    name: 'evaluate_solution',
    description: 'Run the current program through the task evaluator. Returns score, validity, best-so-far, diagnostics, and remaining evaluation budget.',
    parameters: OBJECT,
  },
  {
    name: 'probe_solution',
    description: 'Cheap approximate score of the current program on subsampled data. It does not consume the full evaluation budget; confirm promising variants with evaluate_solution.',
    parameters: OBJECT,
  },
  {
    name: 'finish',
    description: 'End the discovery session and submit the best-scoring program retained by the evaluator.',
    parameters: {
      type: 'object',
      properties: {
        summary: { type: 'string', minLength: 1, description: 'One line describing the final approach and score.' },
      },
      required: ['summary'],
      additionalProperties: false,
    },
    concludesTurn: true,
  },
])

const OUTER_TOOLS = Object.freeze([
  {
    name: 'harness_shell',
    description: 'Inspect the private candidate Cordis harness with one allowed read-only command: pwd, ls, cat, find, or tree. Start with `cat cordis.yml`.',
    parameters: {
      type: 'object',
      properties: { command: { type: 'string', description: 'One allowed read-only command.' } },
      required: ['command'],
      additionalProperties: false,
    },
  },
  {
    name: 'write_harness_file',
    description: 'Create or completely replace cordis.yml or one plugins/*.mjs file in the private candidate harness. Read an existing file before overwriting it.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Relative candidate path.' },
        content: { type: 'string', description: 'Complete UTF-8 file contents.' },
      },
      required: ['path', 'content'],
      additionalProperties: false,
    },
  },
  {
    name: 'edit_harness_file',
    description: 'Modify one inspected candidate file by one exact replacement, or append a small section. Use this for targeted cordis.yml/plugin edits.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string' },
        old_text: { type: 'string' },
        new_text: { type: 'string' },
        append_text: { type: 'string' },
      },
      required: ['path'],
      additionalProperties: false,
    },
  },
  {
    name: 'delete_harness_file',
    description: 'Delete one inspected plugins/*.mjs file after removing its entry from cordis.yml. Core runtime files cannot be deleted.',
    parameters: {
      type: 'object',
      properties: { path: { type: 'string' } },
      required: ['path'],
      additionalProperties: false,
    },
  },
  {
    name: 'validate_harness',
    description: 'Parse, safety-check, and canonically compile the current cordis.yml and plugin sources. Always call this after edits and before submission.',
    parameters: OBJECT,
  },
  {
    name: 'submit_harness',
    description: 'Revalidate and submit the current candidate Cordis harness, then end the proposer session. Submit only after validate_harness succeeds.',
    parameters: OBJECT,
    concludesTurn: true,
  },
])

function integer(value, fallback) {
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback
}

function nonnegativeInteger(value, fallback) {
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : fallback
}

function finite(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function stringArray(value) {
  return Array.isArray(value) ? value.filter(item => typeof item === 'string') : []
}

function truncate(value, limit) {
  const text = typeof value === 'string' ? value : JSON.stringify(value)
  if (!Number.isSafeInteger(limit) || limit < 256 || text.length <= limit) return text
  const head = Math.floor(limit * 0.7)
  const tail = limit - head
  return `${text.slice(0, head)}\n... [SAH truncated ${text.length - limit} chars] ...\n${text.slice(-tail)}`
}

export function apply(ctx, config = {}) {
  const endpoint = process.env.SAH_CORDIS_BRIDGE_URL
  const token = process.env.SAH_CORDIS_BRIDGE_TOKEN
  const role = process.env.SAH_CORDIS_ROLE
  if (!endpoint && !token && !role) return
  if (!endpoint || !token || !['inner', 'outer'].includes(role)) {
    throw new Error('incomplete SAH Cordis bridge environment')
  }

  const maxOutputChars = integer(config.maxOutputChars, role === 'inner' ? 8000 : 6000)
  const call = async (tool, args = {}, signal = undefined, requestId = '') => {
    const response = await fetch(`${endpoint}/v1/tools/${encodeURIComponent(tool)}`, {
      method: 'POST',
      headers: {
        authorization: `Bearer ${token}`,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ arguments: args, request_id: requestId }),
      signal,
    })
    const payload = await response.json().catch(() => ({ ok: false, error: `HTTP ${response.status}` }))
    if (!response.ok || payload.ok !== true) {
      throw new Error(`SAH bridge ${tool} failed: ${payload.error ?? `HTTP ${response.status}`}`)
    }
    return payload.result
  }

  ctx.provide('sahBridge', Object.freeze({ call }))

  const disabledTools = new Set(stringArray(config.disabledTools))
  const toolDescriptions = config.toolDescriptions && typeof config.toolDescriptions === 'object'
    ? config.toolDescriptions
    : {}
  const definitions = (role === 'inner' ? INNER_TOOLS : OUTER_TOOLS)
    .filter(definition => !disabledTools.has(definition.name))
  for (const definition of definitions) {
    ctx.tools.register({
      name: definition.name,
      description: typeof toolDescriptions[definition.name] === 'string'
        ? toolDescriptions[definition.name]
        : definition.description,
      parameters: definition.parameters,
      output: {
        schema: { type: 'string' },
        render: (_args, value) => [{ type: 'text', text: value }],
      },
      async execute(args, exec) {
        const result = await call(definition.name, args, exec.signal, String(exec.callId ?? ''))
        if (definition.concludesTurn) exec.concludeTurn()
        return truncate(result, maxOutputChars)
      },
    })
  }

  // Generated tool lifecycle is observed by the trusted registry pipeline,
  // so a candidate cannot claim participation without an actual dispatch.
  const generatedTools = new Set(stringArray(config.generatedTools))
  if (role === 'inner' && generatedTools.size > 0) {
    ctx.on('tools/pre-execute', async (exec, next) => {
      if (generatedTools.has(exec.name)) {
        await call('__sah_tool_event', { name: exec.name, event: 'invoked' }, exec.signal, String(exec.callId ?? ''))
      }
      return next()
    })
    ctx.on('tools/post-execute', async (exec, result, next) => {
      const decision = await next()
      if (generatedTools.has(exec.name)) {
        await call('__sah_tool_event', {
          name: exec.name,
          event: result.isError ? 'error' : 'completed',
          error: result.isError ? String(result.error?.message ?? result.error ?? 'tool failed') : '',
        }, exec.signal, String(exec.callId ?? ''))
      }
      return decision
    })
  }

  const temperature = finite(
    process.env.SAH_CORDIS_REQUEST_TEMPERATURE ?? config.temperature,
  )
  const configuredMaxTokens = integer(
    process.env.SAH_CORDIS_REQUEST_MAX_TOKENS,
    integer(config.maxTokens, undefined),
  )
  if (temperature !== undefined || configuredMaxTokens !== undefined) {
    ctx.on('agent/request', async (_payload, next) => {
      const current = await next()
      return {
        ...current,
        ...(temperature === undefined ? {} : { temperature }),
        ...(configuredMaxTokens === undefined ? {} : { maxTokens: configuredMaxTokens }),
      }
    })
  }

  const maxIterations = integer(
    process.env.SAH_CORDIS_MAX_ITERATIONS,
    integer(config.maxIterations, role === 'inner' ? 36 : 32),
  )
  const generatedMiddlewares = stringArray(config.generatedMiddlewares)
  const budgetReminderFromLeft = nonnegativeInteger(config.budgetReminderFromLeft, 3)
  const stallAfter = integer(config.stallAfter, 8)
  const maxRestarts = nonnegativeInteger(config.maxRestarts, 2)
  let restarts = 0
  let lastRestartAt = -1
  ctx.on('agent/pre-step', async (payload, next) => {
    if (payload.step > maxIterations) return Promise.resolve({ kind: 'reject' })
    if (role === 'inner') {
      for (const middleware of generatedMiddlewares) {
        await call('__sah_middleware_event', {
          name: middleware, event: 'invoked', iteration: payload.step,
        }, payload.signal, `middleware:${middleware}:${payload.turn}:${payload.step}`)
      }
    }
    const decision = await next()
    if (role !== 'inner' || decision.kind === 'reject') return decision
    let snapshot
    try {
      snapshot = await call('__sah_snapshot', { iteration: payload.step }, payload.signal, `snapshot:${payload.turn}:${payload.step}`)
    } catch (_error) {
      return decision
    }
    const notes = []
    const left = Number(snapshot.evaluations_remaining ?? snapshot.evals_remaining)
    if (Number.isFinite(left) && left <= budgetReminderFromLeft) {
      notes.push(`Evaluation budget reminder: ${left} authoritative evaluation(s) remain. Consolidate the strongest line and avoid unchanged evaluations.`)
    }
    const stalled = Number(snapshot.stalled_evals)
    if (Number.isFinite(stalled) && stalled >= stallAfter && restarts < maxRestarts && stalled !== lastRestartAt) {
      notes.push(`Stall restart ${restarts + 1}/${maxRestarts}: no best-score gain across ${stalled} evaluations. Switch to a structurally different construction family.`)
      restarts += 1
      lastRestartAt = stalled
    }
    if (notes.length === 0) return decision
    const text = notes.join('\n')
    return {
      kind: 'enter',
      messages: [
        ...decision.messages,
        {
          id: `sah-runtime-${payload.turn}-${payload.step}`,
          role: 'user',
          content: [{ type: 'text', text }],
          source: {
            kind: 'plugin', plugin: name, form: 'snapshot',
            sections: [{ name: 'sah-runtime-policy', text }],
          },
        },
      ],
    }
  })
}
