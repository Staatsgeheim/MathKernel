import { describe, expect, it, vi } from 'vitest';
import { HostCommandService } from '../src/host/client';
import { admittedResult, receipt, verifyPages } from '../src/evidence/result';
import { handshakeSchema, type Handshake, type ResultObservation } from '../src/host/contracts';
import { acceptSnapshot, runSchema } from '../src/host/workflow';
const h: Handshake = {
  host_instance_id: 'h',
  workspace_id: 'w',
  mathkernel_version: '1.3.0',
  ui_protocol: 'studio-host/1',
  catalog_revision: 'c1',
  test_host: false,
  authenticated: true,
  features: {
    catalog_read: true,
    authoring_draft: true,
    operation_invoke: false,
    workflow_validate: false,
    workflow_execute: false,
    run_observe: false,
    result_inspect: true,
    artifact_view: false,
    compute_review: false,
    approval_interact: false,
  },
  limits: { control_bytes: 2097152, catalog_page: 25 },
};
const workflowHost: Handshake = {
  ...h,
  features: { ...h.features, workflow_validate: true, workflow_execute: true, run_observe: true },
  extensions: {
    contract: 'studio-workflow/1',
    scopes: ['workflow_outputs'],
    objects: false,
    subworkflows: false,
  },
};
function envelope(payload: unknown, init?: RequestInit, scope = 'h') {
  return new Response(
    JSON.stringify({
      protocol: 'studio-host/1',
      host_instance_id: scope,
      workspace_id: 'w',
      request_id: (init?.headers as Record<string, string>)['X-Studio-Request'],
      observed_at: '2026-09-07T03:00:00Z',
      payload,
    }),
    { headers: { 'content-type': 'application/json' } },
  );
}
const result: ResultObservation = {
  binding: {
    result_ref: 'r',
    source_ref: 'existing',
    source_revision: '1',
    run_ref: null,
    document_id: null,
    draft_revision: null,
    node_id: null,
  },
  admission: 'candidate',
  result: { trust: 'formal', semantic_status: 'proved' },
  claim_trust: {},
  observed_at: '2026-09-07T03:00:00Z',
  scope: { host_instance_id: 'h', workspace_id: 'w' },
};
describe('host observation boundaries', () => {
  it('requires explicit workflow protocol even if a legacy host advertises execution', async () => {
    const f = vi.fn(async (_url: RequestInfo | URL, init?: RequestInit) =>
      envelope({ ...h, features: { ...h.features, workflow_execute: true } }, init),
    );
    const c = new HostCommandService(f);
    await c.connect();
    await expect(c.submit('plan', 'a'.repeat(64), null, 'request')).rejects.toThrow(
      'compatible host',
    );
    expect(f).toHaveBeenCalledTimes(1);
  });
  it('lost submission response sends exactly one mutation and allows only an explicit reconciliation read', async () => {
    const f = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) => {
      if (String(url).endsWith('handshake')) return envelope(workflowHost, init);
      if (String(url).endsWith('workflow/submit'))
        throw new Error('Connection lost after dispatch');
      return envelope(
        {
          client_request_id: 'req-1',
          plan_ref: 'plan',
          outcome: 'accepted',
          run_ref: 'run',
          message: 'Existing request found',
        },
        init,
      );
    });
    const c = new HostCommandService(f);
    await c.connect();
    await expect(c.submit('plan', 'a'.repeat(64), null, 'req-1')).rejects.toThrow('lost');
    expect(f.mock.calls.filter(([, options]) => options?.method === 'POST')).toHaveLength(1);
    expect((await c.reconcile('req-1')).run_ref).toBe('run');
    expect(f.mock.calls.at(-1)![1]!.method).toBe('GET');
  });
  it('preserves newer run facts, rejects changed frozen identities and deduplicates events', () => {
    const run = runSchema.parse({
      run_ref: 'run',
      binding: { document_id: 'doc', draft_revision: 1, document_digest: 'a'.repeat(64) },
      plan_ref: 'plan',
      revision: 2,
      cursor: 'cursor',
      observed_at: '2026-09-07T00:00:00Z',
      execution: 'output ready',
      verification: 'inconclusive',
      artifacts: 'partial',
      resources: 'unknown',
      cost: 'unknown',
      warnings: [],
      attempts: [],
      events: [],
      actions: [],
    });
    expect(acceptSnapshot(run, { ...run, revision: 1, execution: 'running' })).toEqual(run);
    expect(() =>
      acceptSnapshot(run, { ...run, binding: { ...run.binding, draft_revision: 2 } }),
    ).toThrow('identity');
    expect(() => acceptSnapshot(run, { ...run, cost: 'zero' })).toThrow('Conflicting');
    const e = {
      event_id: 'event',
      revision: 3,
      kind: 'progress',
      message: 'Existing event',
      observed_at: run.observed_at,
    };
    expect(acceptSnapshot(run, { ...run, revision: 3, events: [e, e] }).events).toHaveLength(1);
  });
  it('constructing a client triggers no network request', () => {
    const f = vi.fn();
    new HostCommandService(f);
    expect(f).not.toHaveBeenCalled();
  });
  it('rejects incompatible protocol', () =>
    expect(() => handshakeSchema.parse({ ...h, ui_protocol: 'studio-host/9' })).toThrow());
  it('connects and reads catalog with scoped correlation and no workflow calls', async () => {
    const f = vi.fn(async (url: RequestInfo | URL, init?: RequestInit) =>
      envelope(
        String(url).endsWith('handshake')
          ? h
          : { revision: 'c1', entries: [], next_offset: null, total: 0 },
        init,
      ),
    );
    const c = new HostCommandService(f);
    await c.connect();
    expect(await c.catalog()).toEqual([]);
    expect(f).toHaveBeenCalledTimes(2);
    expect(f.mock.calls.every(([, init]) => init?.method === 'GET')).toBe(true);
  });
  it('rejects handshake scope mismatch', async () => {
    const c = new HostCommandService(async (_, init) => envelope(h, init, 'wrong'));
    await expect(c.connect()).rejects.toThrow('identity mismatch');
  });
  it('rejects stale responses after disconnect', async () => {
    let finish: (r: Response) => void = () => {};
    let init: RequestInit | undefined;
    const c = new HostCommandService(async (_, i) => {
      init = i;
      return new Promise<Response>((r) => (finish = r));
    });
    const p = c.connect();
    c.disconnect();
    finish(envelope(h, init));
    await expect(p).rejects.toThrow('Stale');
  });
  it('rejects catalog drift and nonadvancing pagination', async () => {
    const f = async (url: RequestInfo | URL, init?: RequestInit) =>
      envelope(
        String(url).endsWith('handshake')
          ? h
          : { revision: 'c2', entries: [], next_offset: 0, total: 2 },
        init,
      );
    const c = new HostCommandService(f);
    await c.connect();
    await expect(c.catalog()).rejects.toThrow('revision changed');
  });
  it('does not retry the connection mutation after a lost acknowledgement', async () => {
    const f = vi.fn(async () => {
      throw new Error('lost acknowledgement');
    });
    const c = new HostCommandService(f);
    await expect(c.connect('one-time')).rejects.toThrow('lost');
    expect(f).toHaveBeenCalledTimes(1);
    expect(f.mock.calls).toHaveLength(1);
  });
  it('clears session authority on 403', async () => {
    const c = new HostCommandService(async (url, init) =>
      String(url).endsWith('handshake') ? envelope(h, init) : new Response('', { status: 403 }),
    );
    await c.connect();
    await expect(c.catalog()).rejects.toThrow();
    await expect(c.results()).rejects.toThrow('unavailable');
  });
  it('caps streamed response bytes', async () => {
    const c = new HostCommandService(
      async () =>
        new Response(' '.repeat(2097153), { headers: { 'content-type': 'application/json' } }),
    );
    await expect(c.connect()).rejects.toThrow('control budget');
  });
  it('never admits candidate or test-host labels', () => {
    expect(admittedResult(result)).toBeNull();
    expect(admittedResult({ ...result, admission: 'test_fixture' })).toBeNull();
  });
  it('recognizes wrapped and bare receipts before looking at trust', () => {
    const data = { truncated: true, resource_id: 'r', size_bytes: 100, original_trust: 'formal' };
    expect(receipt(data)?.resource_id).toBe('r');
    expect(
      admittedResult({ ...result, admission: 'host', result: { trust: 'formal', data } }),
    ).toBeNull();
  });
  it('verifies full resource bytes and rejects corruption or partial reconstruction', async () => {
    const text = '{"value":"9007199254740993"}';
    const digest = [
      ...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))),
    ]
      .map((v) => v.toString(16).padStart(2, '0'))
      .join('');
    expect(await verifyPages(text, text.length, digest)).toEqual({ value: '9007199254740993' });
    await expect(verifyPages(text, text.length + 1, digest)).rejects.toThrow('Partial');
    await expect(verifyPages(text, text.length, '0'.repeat(64))).rejects.toThrow('integrity');
  });
});
