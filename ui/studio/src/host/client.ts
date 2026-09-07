import { randomId } from '../security/identity';
import { z } from 'zod';
import { CONTROL_BYTES, parseJson } from '../security/json';
import { DOCUMENT_BYTES } from '../security/json';
import {
  validationSchema,
  planSchema,
  challengeSchema,
  approvalSchema,
  submissionSchema,
  runSchema,
  runsSchema,
  objectsSchema,
  subworkflowSchema,
  type Frozen,
} from './workflow';
import type { StudioDocument } from '../editor/document';
import {
  catalogSchema,
  envelopeSchema,
  handshakeSchema,
  pageSchema,
  resultsSchema,
  resultSchema,
  type Handshake,
  type Operation,
  type ResultObservation,
  type Scope,
} from './contracts';

export class HostError extends Error {
  constructor(
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}
async function boundedResponse(response: Response, limit: number): Promise<unknown> {
  if (!response.headers.get('content-type')?.startsWith('application/json'))
    throw new HostError('Host returned an unexpected media type.');
  const reader = response.body?.getReader();
  if (!reader) throw new HostError('Host response has no body.');
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const r = await reader.read();
      if (r.done) break;
      size += r.value.byteLength;
      if (size > limit) throw new HostError('Host response exceeds the control budget.');
      chunks.push(r.value);
    }
  } catch (error) {
    await reader.cancel();
    throw error;
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return parseJson(new TextDecoder('utf-8', { fatal: true }).decode(bytes), limit);
}
/** Only this named service performs network actions. No scheduler and no mutation retry. */
export class HostCommandService {
  private generation = 0;
  private abort = new AbortController();
  private scope: Scope | null = null;
  private handshakeValue: Handshake | null = null;
  constructor(private fetcher: typeof fetch = (input, init) => fetch(input, init)) {}
  disconnect(): void {
    this.generation++;
    this.abort.abort();
    this.abort = new AbortController();
    this.scope = null;
    this.handshakeValue = null;
  }
  private async get<T>(
    path: string,
    schema: z.ZodType<T>,
    action?: { body: unknown; requestId?: string },
  ): Promise<{ payload: T; observed_at: string }> {
    const generation = this.generation,
      requestId = action?.requestId ?? randomId();
    const body = action ? JSON.stringify(action.body) : undefined;
    if (body && new TextEncoder().encode(body).length > DOCUMENT_BYTES + 8192)
      throw new HostError('Command exceeds the document budget.');
    const response = await this.fetcher(`/studio/api/${path}`, {
      method: action ? 'POST' : 'GET',
      body,
      credentials: 'same-origin',
      cache: 'no-store',
      redirect: 'error',
      signal: AbortSignal.any([this.abort.signal, AbortSignal.timeout(15000)]),
      headers: {
        'X-Studio-Request': requestId,
        ...(action ? { 'Content-Type': 'application/json', 'X-Studio-Action': path } : {}),
      },
    });
    if (generation !== this.generation) throw new HostError('Stale host response discarded.');
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) this.disconnect();
      throw new HostError(
        response.status === 401
          ? 'Session expired or not connected.'
          : response.status === 404
            ? 'Reference unavailable or expired.'
            : 'Host request was rejected.',
        response.status,
      );
    }
    const e = envelopeSchema.parse(
      await boundedResponse(response, this.handshakeValue?.limits.control_bytes ?? CONTROL_BYTES),
    );
    if (generation !== this.generation || e.request_id !== requestId)
      throw new HostError('Stale or mismatched host response.');
    if (
      this.scope &&
      (e.host_instance_id !== this.scope.host_instance_id ||
        e.workspace_id !== this.scope.workspace_id)
    )
      throw new HostError('Host/workspace scope mismatch.');
    const payload = schema.parse(e.payload);
    if (path === 'handshake') {
      const h = handshakeSchema.parse(payload);
      if (h.host_instance_id !== e.host_instance_id || h.workspace_id !== e.workspace_id)
        throw new HostError('Handshake identity mismatch.');
    }
    return { payload, observed_at: e.observed_at };
  }
  async connect(code?: string): Promise<Handshake> {
    this.disconnect();
    const generation = this.generation;
    if (code) {
      const response = await this.fetcher('/studio/api/session', {
        method: 'POST',
        credentials: 'same-origin',
        cache: 'no-store',
        redirect: 'error',
        signal: AbortSignal.any([this.abort.signal, AbortSignal.timeout(15000)]),
        headers: { 'Content-Type': 'application/json', 'X-Studio-Action': 'connect' },
        body: JSON.stringify({ code }),
      });
      if (generation !== this.generation) throw new HostError('Connection superseded.');
      if (!response.ok)
        throw new HostError('Connection code rejected or expired.', response.status);
    }
    const { payload } = await this.get('handshake', handshakeSchema);
    if (generation !== this.generation) throw new HostError('Connection superseded.');
    this.scope = { host_instance_id: payload.host_instance_id, workspace_id: payload.workspace_id };
    this.handshakeValue = payload;
    return payload;
  }
  async catalog(): Promise<Operation[]> {
    if (!this.handshakeValue?.features.catalog_read)
      throw new HostError('Catalog reading is unavailable.');
    const entries: Operation[] = [];
    let offset = 0;
    while (true) {
      const { payload } = await this.get(`catalog?offset=${offset}`, catalogSchema);
      if (payload.revision !== this.handshakeValue?.catalog_revision)
        throw new HostError('Catalog revision changed. Reconnect to review updated metadata.');
      entries.push(...payload.entries);
      if (
        entries.length > 20000 ||
        new Set(entries.map((e) => e.descriptor_id)).size !== entries.length
      )
        throw new HostError('Invalid catalog identity or size.');
      if (payload.next_offset === null) {
        if (entries.length !== payload.total) throw new HostError('Incomplete catalog response.');
        break;
      }
      if (payload.next_offset <= offset || payload.next_offset !== entries.length)
        throw new HostError('Catalog paging did not advance.');
      offset = payload.next_offset;
    }
    return entries;
  }
  async results(offset = 0) {
    if (!this.handshakeValue?.features.result_inspect)
      throw new HostError('Result inspection is unavailable.');
    return (await this.get(`results?offset=${offset}`, resultsSchema)).payload;
  }
  async result(ref: string): Promise<ResultObservation> {
    if (!this.handshakeValue?.features.result_inspect || !this.scope)
      throw new HostError('Result inspection is unavailable.');
    const { payload, observed_at } = await this.get(
      `results/${encodeURIComponent(ref)}`,
      resultSchema,
    );
    if (payload.binding.result_ref !== ref) throw new HostError('Result reference mismatch.');
    return { ...payload, observed_at, scope: { ...this.scope } };
  }
  async resultPage(ref: string, resource: string, offset: number) {
    if (!this.handshakeValue?.features.result_inspect)
      throw new HostError('Result inspection is unavailable.');
    const { payload } = await this.get(
      `results/${encodeURIComponent(ref)}/page?offset=${offset}`,
      pageSchema,
    );
    if (
      payload.resource_id !== resource ||
      payload.offset !== offset ||
      (payload.next_offset !== null && payload.next_offset !== offset + payload.content.length)
    )
      throw new HostError('Result page identity/offset mismatch.');
    return payload;
  }
  async resultAdmission(ref: string): Promise<ResultObservation> {
    if (!this.handshakeValue?.features.result_inspect || !this.scope)
      throw new HostError('Result inspection unavailable.');
    const { payload, observed_at } = await this.get(
      `results/${encodeURIComponent(ref)}/admission`,
      resultSchema,
    );
    if (payload.binding.result_ref !== ref) throw new HostError('Admission reference mismatch.');
    return { ...payload, observed_at, scope: { ...this.scope } };
  }
  async revokeSession() {
    const response = await this.fetcher('/studio/api/session/disconnect', {
      method: 'POST',
      credentials: 'same-origin',
      cache: 'no-store',
      redirect: 'error',
      headers: { 'X-Studio-Action': 'disconnect' },
      signal: AbortSignal.timeout(15000),
    });
    if (!response.ok) throw new HostError('Session revocation was not confirmed.', response.status);
    this.disconnect();
  }
  private requireWorkflow(feature: keyof Handshake['features']) {
    if (
      !this.handshakeValue?.features[feature] ||
      this.handshakeValue.extensions?.contract !== 'studio-workflow/1'
    )
      throw new HostError('A compatible host service is unavailable.');
  }
  async validate(document: StudioDocument, binding: Frozen) {
    this.requireWorkflow('workflow_validate');
    return (await this.get('workflow/validate', validationSchema, { body: { document, binding } }))
      .payload;
  }
  async plan(validation_ref: string, binding: Frozen, scope: string, selected_nodes: string[]) {
    this.requireWorkflow('workflow_execute');
    return (
      await this.get('workflow/plan', planSchema, {
        body: { validation_ref, binding, scope, selected_nodes },
      })
    ).payload;
  }
  async challenge(plan_ref: string, plan_digest: string) {
    this.requireWorkflow('approval_interact');
    return (
      await this.get('approval/challenge', challengeSchema, { body: { plan_ref, plan_digest } })
    ).payload;
  }
  async confirm(
    challenge_ref: string,
    plan_ref: string,
    plan_digest: string,
    decision: 'approve' | 'deny',
  ) {
    this.requireWorkflow('approval_interact');
    return (
      await this.get('approval/confirm', approvalSchema, {
        body: { challenge_ref, plan_ref, plan_digest, decision },
      })
    ).payload;
  }
  async submit(
    plan_ref: string,
    plan_digest: string,
    authority_ref: string | null,
    client_request_id: string,
  ) {
    this.requireWorkflow('workflow_execute');
    return (
      await this.get('workflow/submit', submissionSchema, {
        body: { plan_ref, plan_digest, authority_ref, client_request_id },
        requestId: client_request_id,
      })
    ).payload;
  }
  async reconcile(client_request_id: string) {
    this.requireWorkflow('run_observe');
    return (await this.get(`requests/${encodeURIComponent(client_request_id)}`, submissionSchema))
      .payload;
  }
  async run(ref: string) {
    this.requireWorkflow('run_observe');
    const value = (await this.get(`runs/${encodeURIComponent(ref)}`, runSchema)).payload;
    if (value.run_ref !== ref) throw new HostError('Run reference mismatch.');
    return value;
  }
  async runs(offset = 0) {
    this.requireWorkflow('run_observe');
    return (await this.get(`runs?offset=${offset}`, runsSchema)).payload;
  }
  async runAction(
    run_ref: string,
    revision: number,
    action: 'cancel' | 'retry' | 'reverify' | 'resume',
    client_request_id: string,
  ) {
    this.requireWorkflow('run_observe');
    return (
      await this.get('runs/action', submissionSchema, {
        body: { run_ref, revision, action, client_request_id },
        requestId: client_request_id,
      })
    ).payload;
  }
  async objects(offset = 0) {
    if (!this.handshakeValue?.extensions?.objects)
      throw new HostError('Object listing unavailable.');
    return (await this.get(`objects?offset=${offset}`, objectsSchema)).payload;
  }
  async subworkflow(ref: string) {
    if (!this.handshakeValue?.extensions?.subworkflows)
      throw new HostError('Subworkflow navigation unavailable.');
    const value = (await this.get(`subworkflows/${encodeURIComponent(ref)}`, subworkflowSchema))
      .payload;
    if (value.reference !== ref) throw new HostError('Subworkflow reference mismatch.');
    return value;
  }
}
