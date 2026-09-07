import { z } from '../security/schema';
import { DOCUMENT_BYTES, parseJson } from '../security/json';

export const id = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9][A-Za-z0-9_.:-]*$/)
  .refine((v) => !['constructor', 'prototype', '__proto__'].includes(v));
const revision = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER);
export const position = z.strictObject({
  x: z.number().min(-1e6).max(1e6),
  y: z.number().min(-1e6).max(1e6),
});
const draft = z.strictObject({
  encoding: z.enum(['text', 'json_text']),
  text: z.string().max(1048576),
});
export const nodeSchema = z.strictObject({
  id,
  kind: z.enum(['operation', 'object_binding', 'subworkflow', 'unresolved']),
  label: z.string().min(1).max(8192),
  operation_ref: id.nullable(),
  operation_version: z.string().max(128).nullable(),
  schema_digest: z
    .string()
    .regex(/^[0-9a-f]{64}$/)
    .nullable(),
  parameter_drafts: z.record(id, draft).refine((v) => Object.keys(v).length <= 256),
  input_bindings: z
    .record(
      id,
      z.strictObject({
        host_instance_id: id,
        workspace_id: id,
        object_id: id,
        resolution: z.literal('unresolved'),
      }),
    )
    .refine((v) => Object.keys(v).length <= 256),
});
export const edgeSchema = z.strictObject({
  id,
  source_node: id,
  source_port: id,
  target_node: id,
  target_port: id,
});
export const documentSchema = z.strictObject({
  schema: z.literal('mk.studio/1'),
  identity: z.strictObject({ document_id: id, title: z.string().min(1).max(8192) }),
  authoring: z.strictObject({
    revision,
    nodes: z.array(nodeSchema).max(5000),
    edges: z.array(edgeSchema).max(10000),
    desired_outputs: z.array(z.strictObject({ node_id: id, port_id: id })).max(5000),
  }),
  presentation: z.strictObject({
    revision,
    node_positions: z.record(id, position).refine((v) => Object.keys(v).length <= 5000),
    groups: z
      .array(
        z.strictObject({ id, title: z.string().min(1).max(8192), members: z.array(id).max(5000) }),
      )
      .max(5000),
    viewport: z.strictObject({ x: z.number(), y: z.number(), zoom: z.number().min(0.05).max(8) }),
  }),
  host_binding: z
    .strictObject({
      host_instance_id: id,
      workspace_id: id,
      workflow_ref: id,
      workflow_revision: id,
    })
    .nullable(),
});
export type StudioDocument = z.infer<typeof documentSchema>;
export type DraftNode = z.infer<typeof nodeSchema>;
export type DraftEdge = z.infer<typeof edgeSchema>;
export type Position = z.infer<typeof position>;
export const newId = () => crypto.randomUUID();
export function emptyDocument(): StudioDocument {
  return {
    schema: 'mk.studio/1',
    identity: { document_id: newId(), title: 'Untitled experiment' },
    authoring: { revision: 0, nodes: [], edges: [], desired_outputs: [] },
    presentation: {
      revision: 0,
      node_positions: {},
      groups: [],
      viewport: { x: 0, y: 0, zoom: 1 },
    },
    host_binding: null,
  };
}
export function validateDocument(value: unknown): StudioDocument {
  const d = documentSchema.parse(value);
  for (const list of [d.authoring.nodes, d.authoring.edges, d.presentation.groups]) {
    if (new Set(list.map((v) => v.id)).size !== list.length)
      throw new Error('Duplicate entity ID.');
  }
  const nodes = new Set(d.authoring.nodes.map((n) => n.id));
  for (const e of d.authoring.edges)
    if (!nodes.has(e.source_node) || !nodes.has(e.target_node)) throw new Error('Dangling edge.');
  for (const out of d.authoring.desired_outputs)
    if (!nodes.has(out.node_id)) throw new Error('Dangling desired output.');
  for (const key of Object.keys(d.presentation.node_positions))
    if (!nodes.has(key)) throw new Error('Dangling node position.');
  for (const g of d.presentation.groups)
    if (g.members.some((n) => !nodes.has(n))) throw new Error('Dangling group member.');
  for (const n of d.authoring.nodes) {
    const bytes = Object.values(n.parameter_drafts).reduce(
      (s, v) => s + new TextEncoder().encode(v.text).length,
      0,
    );
    if (bytes > 1048576) throw new Error('Node exceeds 1 MiB inline parameter limit.');
  }
  return d;
}
export function importDocument(text: string): StudioDocument {
  return validateDocument(parseJson(text));
}
export function serializeDocument(d: StudioDocument): string {
  const text = JSON.stringify(validateDocument(d), null, 2);
  if (new TextEncoder().encode(text).length > DOCUMENT_BYTES)
    throw new Error('Document exceeds 10 MiB.');
  return text;
}

export function makeInput(kind: 'integer' | 'rational' | 'real'): DraftNode {
  const value =
    kind === 'rational'
      ? { kind, numerator: '1', denominator: '3' }
      : kind === 'real'
        ? { kind, value: '0.1', precision: 53 }
        : { kind, value: '9007199254740993' };
  return {
    id: newId(),
    kind: 'object_binding',
    label: `${kind === 'real' ? 'Numerical' : 'Exact'} ${kind} input`,
    operation_ref: null,
    operation_version: null,
    schema_digest: null,
    parameter_drafts: { value: { encoding: 'json_text', text: JSON.stringify(value, null, 2) } },
    input_bindings: {},
  };
}
