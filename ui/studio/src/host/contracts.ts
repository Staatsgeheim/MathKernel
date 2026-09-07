import { z } from '../security/schema';
import { id } from '../editor/document';
export const PROTOCOL = 'studio-host/1';
const text = z.string().max(8192);
export const featuresSchema = z.strictObject({
  catalog_read: z.boolean(),
  authoring_draft: z.boolean(),
  operation_invoke: z.boolean(),
  workflow_validate: z.boolean(),
  workflow_execute: z.boolean(),
  run_observe: z.boolean(),
  result_inspect: z.boolean(),
  artifact_view: z.boolean(),
  compute_review: z.boolean(),
  approval_interact: z.boolean(),
});
export const handshakeSchema = z.strictObject({
  host_instance_id: id,
  workspace_id: id,
  mathkernel_version: z.string().max(128),
  ui_protocol: z.literal(PROTOCOL),
  catalog_revision: id,
  test_host: z.boolean(),
  features: featuresSchema,
  authenticated: z.literal(true),
  limits: z.strictObject({
    control_bytes: z.number().int().min(1000).max(2097152),
    catalog_page: z.number().int().min(1).max(100),
  }),
});
export type Handshake = z.infer<typeof handshakeSchema>;
export const portSchema = z.strictObject({
  id,
  direction: z.enum(['input', 'output']),
  label: text,
  type_ref: text.nullable(),
  cardinality: z.enum(['one', 'many', 'unknown']),
  required: z.boolean().nullable(),
  shape: z
    .array(z.union([z.number().int().nonnegative(), z.string().max(128), z.null()]))
    .max(32)
    .nullable(),
});
export const operationSchema = z.strictObject({
  descriptor_id: id,
  operation_ref: id,
  operation_version: z.string().max(128),
  schema_digest: z.string().regex(/^[0-9a-f]{64}$/),
  title: text,
  domain: text,
  description: text,
  availability: z.enum(['available', 'unavailable', 'experimental', 'unknown']),
  composition: z.enum(['complete', 'partial', 'documentation_only']),
  input_ports: z.array(portSchema).max(256),
  output_ports: z.array(portSchema).max(256),
  input_types: z.array(text).max(256),
  output_types: z.array(text).max(256),
  engines: z.array(text).max(256),
  trust_levels: z.array(text).max(32),
  verification_methods: z.array(text).max(256),
  parameter_schema: z.record(z.string(), z.unknown()),
  coverage: text,
});
export type Operation = z.infer<typeof operationSchema>;
export type Port = z.infer<typeof portSchema>;
export const catalogSchema = z.strictObject({
  revision: id,
  entries: z.array(operationSchema).max(100),
  next_offset: z.number().int().nonnegative().nullable(),
  total: z.number().int().nonnegative().max(20000),
});
export const envelopeSchema = z.strictObject({
  protocol: z.literal(PROTOCOL),
  host_instance_id: id,
  workspace_id: id,
  request_id: id,
  observed_at: z.iso.datetime(),
  payload: z.unknown(),
});
export type Scope = Pick<Handshake, 'host_instance_id' | 'workspace_id'>;
export const TRUST = [
  'formal',
  'exact',
  'symbolic',
  'interval_certified',
  'numeric_high_precision',
  'numeric',
  'empirical',
  'heuristic',
  'unknown',
] as const;
export const RESULT_STATUS = [
  'proved',
  'verified_exact',
  'verified_symbolic',
  'refuted',
  'certified',
  'verified_numeric',
  'numeric',
  'empirical',
  'candidate',
  'unknown',
  'does_not_exist',
  'undefined',
  'infeasible',
  'unbounded',
  'unsupported',
  'error',
] as const;
export const resultBindingSchema = z.strictObject({
  result_ref: id,
  source_ref: id,
  source_revision: id,
  run_ref: id.nullable(),
  document_id: id.nullable(),
  draft_revision: z.number().int().nonnegative().nullable(),
  node_id: id.nullable(),
});
const evidenceItem = z.record(z.string(), z.unknown());
const bundleSchema = z.object({
  computation: z.array(evidenceItem).max(1000),
  proof: z.array(evidenceItem).max(1000),
  certificate: z.array(evidenceItem).max(1000),
  numerical: z.array(evidenceItem).max(1000),
  model: z.array(evidenceItem).max(1000),
  empirical: z.array(evidenceItem).max(1000),
  justified_trust: z.enum(TRUST).nullable(),
});
export const mathResultSchema = z
  .object({
    ok: z.boolean(),
    status: z.enum(['verified', 'refuted', 'candidate', 'unknown', 'error', 'ok']),
    trust: z.enum(TRUST),
    semantic_status: z.enum(RESULT_STATUS).nullable(),
    data: z.record(z.string(), z.unknown()),
    engine: z.string().nullable(),
    assumptions_used: z.array(text).max(1000),
    side_conditions: z.array(text).max(1000),
    warnings: z.array(text).max(1000),
    errors: z.array(text).max(1000),
    claim_evidence: z.record(z.string().max(128), bundleSchema),
    arithmetic_transition: z.record(z.string(), text),
    derivation: z.array(z.unknown()).max(1000),
    evidence_bundle: bundleSchema,
    engine_versions: z.record(z.string(), text),
    mathkernel_version: text.nullable(),
  })
  .passthrough();
export const resultSchema = z.strictObject({
  binding: resultBindingSchema,
  admission: z.enum(['host', 'candidate', 'test_fixture']),
  result: z.unknown(),
  claim_trust: z.record(z.string().max(128), z.enum(TRUST)),
});
export type ResultObservation = z.infer<typeof resultSchema> & {
  observed_at: string;
  scope: Scope;
};
export const resultsSchema = z.strictObject({
  results: z.array(resultBindingSchema).max(100),
  next_offset: z.number().int().nonnegative().nullable(),
});
export const pageSchema = z.strictObject({
  resource_id: id,
  content: z.string().max(1048576),
  offset: z.number().int().nonnegative(),
  next_offset: z.number().int().nonnegative().nullable(),
  total_bytes: z.number().int().nonnegative(),
  sha256: z.string().regex(/^[0-9a-f]{64}$/),
  media_type: z.literal('application/json'),
});
