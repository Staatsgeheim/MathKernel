import { z } from '../security/schema';
import { id } from '../editor/document';
const text = z.string().max(8192),
  revision = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const digest = z.string().regex(/^[0-9a-f]{64}$/);
export const workflowExtensions = z.strictObject({
  contract: z.literal('studio-workflow/1'),
  scopes: z.array(z.enum(['workflow_outputs', 'selected_subgraph', 'standalone_operation'])).max(3),
  objects: z.boolean(),
  subworkflows: z.boolean(),
});
export const frozenSchema = z.strictObject({
  document_id: id,
  draft_revision: revision,
  document_digest: digest,
});
export type Frozen = z.infer<typeof frozenSchema>;
export const validationSchema = z.strictObject({
  binding: frozenSchema,
  validation_ref: id,
  valid: z.boolean(),
  diagnostics: z
    .array(
      z.strictObject({
        code: id,
        severity: z.enum(['error', 'warning', 'information', 'unresolved']),
        message: text,
        node_id: id.nullable(),
        field: text.nullable(),
      }),
    )
    .max(2000),
});
export const planSchema = z.strictObject({
  binding: frozenSchema,
  plan_ref: id,
  digest,
  expires_at: z.iso.datetime(),
  scope: z.enum(['workflow_outputs', 'selected_subgraph', 'standalone_operation']),
  selected_nodes: z.array(id).max(5000),
  outputs: z.array(text).max(5000),
  operations: z
    .array(
      z.strictObject({ node_id: id, operation_ref: id, version: text, engine: text, target: text }),
    )
    .max(5000),
  assumptions: z.array(text).max(1000),
  arithmetic: z.array(text).max(1000),
  evidence_requirements: z.array(text).max(1000),
  resources: z
    .array(z.strictObject({ name: text, amount: text.nullable(), unit: text, source: text }))
    .max(100),
  cost: z.strictObject({
    amount: z
      .string()
      .regex(/^\d+(\.\d+)?$/)
      .max(128)
      .nullable(),
    currency: z.string().max(16).nullable(),
    source: text,
    observed_at: z.iso.datetime(),
    uncertainty: text,
    exclusions: z.array(text).max(100),
  }),
  exports: z
    .array(
      z.strictObject({
        input_ref: id,
        classification: text,
        destination: text,
        bytes: z.string().regex(/^\d+$/).max(128).nullable(),
      }),
    )
    .max(1000),
  alternatives: z
    .array(z.strictObject({ target: text, accepted: z.boolean(), reason: text }))
    .max(100),
  authorization_required: z.boolean(),
  policy_denied: z.boolean(),
  warnings: z.array(text).max(1000),
});
export type Plan = z.infer<typeof planSchema>;
export const challengeSchema = z.strictObject({
  challenge_ref: id,
  plan_ref: id,
  plan_digest: digest,
  expires_at: z.iso.datetime(),
  disclosures: z.array(text).max(1000),
  can_confirm: z.boolean(),
});
export const approvalSchema = z.strictObject({
  plan_ref: id,
  plan_digest: digest,
  authority_ref: id.nullable(),
  decision: z.enum(['approved', 'denied']),
  expires_at: z.iso.datetime(),
});
export const submissionSchema = z.strictObject({
  client_request_id: id,
  plan_ref: id,
  outcome: z.enum(['accepted', 'rejected', 'unknown']),
  run_ref: id.nullable(),
  message: text,
});
export type Submission = z.infer<typeof submissionSchema>;
export const runSchema = z.strictObject({
  run_ref: id,
  binding: frozenSchema,
  plan_ref: id,
  revision,
  cursor: id,
  observed_at: z.iso.datetime(),
  execution: text,
  verification: text,
  artifacts: text,
  resources: text,
  cost: text,
  warnings: z.array(text).max(1000),
  attempts: z
    .array(
      z.strictObject({
        node_id: id,
        attempt_id: id,
        revision,
        state: text,
        result_ref: id.nullable(),
        progress: z.number().min(0).max(100).nullable(),
        progress_kind: z.enum(['measured', 'estimated', 'indeterminate']),
        observed_at: z.iso.datetime(),
      }),
    )
    .max(2000),
  events: z
    .array(
      z.strictObject({
        event_id: id,
        revision,
        kind: text,
        message: text,
        observed_at: z.iso.datetime(),
      }),
    )
    .max(500),
  actions: z.array(z.enum(['cancel', 'retry', 'reverify', 'resume'])).max(4),
});
export type Run = z.infer<typeof runSchema>;
export const runsSchema = z.strictObject({
  runs: z
    .array(
      z.strictObject({ run_ref: id, document_id: id, draft_revision: revision, execution: text }),
    )
    .max(100),
  next_offset: revision.nullable(),
});
export const objectsSchema = z.strictObject({
  objects: z
    .array(
      z.strictObject({
        object_id: id,
        revision: id,
        type_ref: text,
        summary: text,
        shape: z
          .array(z.union([revision, text, z.null()]))
          .max(32)
          .nullable(),
      }),
    )
    .max(100),
  next_offset: revision.nullable(),
});
export const subworkflowSchema = z.strictObject({
  reference: id,
  revision: id,
  digest,
  title: text,
  read_only: z.boolean(),
  required_capabilities: z.array(text).max(100),
  boundary: z
    .array(z.strictObject({ external_port: id, internal_node: id, internal_port: id }))
    .max(1000),
  description: text,
});

export async function freezeDocument(
  document: import('../editor/document').StudioDocument,
): Promise<Frozen> {
  if (!crypto.subtle)
    throw new Error(
      'Plan binding requires a secure browser context. Open the operator-provided loopback or HTTPS address.',
    );
  // Labels and layout do not redefine the mathematical revision.
  const bytes = new TextEncoder().encode(
    JSON.stringify({
      document_id: document.identity.document_id,
      ...document.authoring,
      nodes: document.authoring.nodes.map((n) => ({ ...n, label: '' })),
    }),
  );
  const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return {
    document_id: document.identity.document_id,
    draft_revision: document.authoring.revision,
    document_digest: Array.from(hash, (b) => b.toString(16).padStart(2, '0')).join(''),
  };
}
export function sameBinding(a: Frozen, b: Frozen) {
  return (
    a.document_id === b.document_id &&
    a.draft_revision === b.draft_revision &&
    a.document_digest === b.document_digest
  );
}
export function acceptSnapshot(previous: Run | null, incoming: Run): Run {
  // A snapshot is authoritative only if its own entity facts are internally consistent.
  const attempts = new Map<string, Run['attempts'][number]>();
  for (const attempt of incoming.attempts) {
    if (attempts.has(attempt.attempt_id)) throw new Error('Duplicate attempt identity.');
    attempts.set(attempt.attempt_id, attempt);
  }
  const events = new Map<string, Run['events'][number]>();
  for (const event of incoming.events) {
    const duplicate = events.get(event.event_id);
    if (duplicate && JSON.stringify(duplicate) !== JSON.stringify(event))
      throw new Error('Conflicting event identity requires host resynchronization.');
    events.set(event.event_id, event);
  }
  incoming = { ...incoming, events: [...events.values()] };
  if (!previous) return incoming;
  if (
    previous.run_ref !== incoming.run_ref ||
    !sameBinding(previous.binding, incoming.binding) ||
    previous.plan_ref !== incoming.plan_ref
  )
    throw new Error('Run identity or frozen binding changed.');
  if (incoming.revision < previous.revision) return previous;
  if (
    incoming.revision === previous.revision &&
    JSON.stringify(incoming) !== JSON.stringify(previous)
  ) {
    // A timestamp may be refreshed; facts at a fixed entity revision cannot change.
    const facts = (r: Run) => JSON.stringify({ ...r, observed_at: '' });
    if (facts(incoming) !== facts(previous))
      throw new Error('Conflicting run observations require host resynchronization.');
  }
  for (const attempt of incoming.attempts) {
    const old = previous.attempts.find((a) => a.attempt_id === attempt.attempt_id);
    if (old && (old.node_id !== attempt.node_id || attempt.revision < old.revision))
      throw new Error('Attempt identity/revision regressed.');
    if (old && old.revision === attempt.revision && JSON.stringify(old) !== JSON.stringify(attempt))
      throw new Error('Conflicting attempt facts at a fixed revision.');
  }
  for (const event of incoming.events) {
    const old = previous.events.find((e) => e.event_id === event.event_id);
    if (old && JSON.stringify(old) !== JSON.stringify(event))
      throw new Error('Conflicting event facts across snapshots.');
  }
  return {
    ...incoming,
    events: [...new Map(incoming.events.map((e) => [e.event_id, e])).values()].slice(-500),
  };
}
