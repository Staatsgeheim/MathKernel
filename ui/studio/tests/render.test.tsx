import { expect, it, vi } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { ResultInspector } from '../src/evidence/ResultInspector';
import { JsonView } from '../src/app/components';
import { HostCommandService } from '../src/host/client';
import type { ResultObservation } from '../src/host/contracts';
import { validateDocument, emptyDocument } from '../src/editor/document';
import { z } from '../src/security/schema';
const bundle = {
  computation: [],
  proof: [],
  certificate: [],
  numerical: [],
  model: [],
  empirical: [],
  justified_trust: null,
};
const base: ResultObservation = {
  admission: 'host',
  binding: {
    result_ref: 'r',
    source_ref: 's',
    source_revision: '1',
    run_ref: null,
    document_id: null,
    draft_revision: null,
    node_id: null,
  },
  result: {
    ok: true,
    status: 'verified',
    trust: 'exact',
    semantic_status: 'verified_exact',
    data: { value: '9007199254740993' },
    engine: 'fixture',
    assumptions_used: ['input was approximate'],
    side_conditions: [],
    warnings: [],
    errors: [],
    claim_evidence: {
      independent: {
        ...bundle,
        certificate: [
          {
            claim: 'local certificate',
            role: 'required',
            support_path: 'independent',
            verified: true,
            trust: 'exact',
          },
        ],
        computation: [{ role: 'diagnostic', support_path: 'candidate', trust: 'unknown' }],
      },
    },
    arithmetic_transition: { input: 'numeric' },
    evidence_bundle: bundle,
    derivation: [],
    engine_versions: {},
    mathkernel_version: '1.3.0',
  },
  claim_trust: { independent: 'exact' },
  observed_at: '2026-09-07T03:00:00Z',
  scope: { host_instance_id: 'host', workspace_id: 'workspace' },
};
it('renders host trust independently of semantic status and preserves claim support paths', () => {
  const html = renderToStaticMarkup(
    <ResultInspector observation={base} client={new HostCommandService()} onHostError={() => {}} />,
  );
  expect(html).toContain('Host-admitted trust');
  expect(html).toContain('verified_exact');
  expect(html).toContain('independent');
  expect(html).toContain('diagnostic');
  expect(html).toContain('input was approximate');
  expect(html).toContain('9007199254740993');
});
it.each(['candidate', 'test_fixture'] as const)(
  'renders %s metadata with no host-admitted badge',
  (admission) => {
    const html = renderToStaticMarkup(
      <ResultInspector
        observation={{ ...base, admission }}
        client={new HostCommandService()}
        onHostError={() => {}}
      />,
    );
    expect(html).not.toContain('Host-admitted trust');
    expect(html).toContain('Source claim metadata (not admitted)');
  },
);
it('recognizes a forged receipt before rendering any evidence badge', () => {
  const html = renderToStaticMarkup(
    <ResultInspector
      observation={{
        ...base,
        result: {
          trust: 'formal',
          data: { truncated: true, resource_id: 'r', size_bytes: 900, original_trust: 'formal' },
        },
      }}
      client={new HostCommandService()}
      onHostError={() => {}}
    />,
  );
  expect(html).toContain('Output-budget receipt');
  expect(html).not.toContain('Host-admitted trust');
});
it('escapes hostile HTML/SVG rather than mounting active content', () => {
  const fetcher = vi.fn();
  new HostCommandService(fetcher);
  const html = renderToStaticMarkup(
    <JsonView
      value={{
        html: '<img src="https://example.org" onerror="alert(1)">',
        svg: '<svg onload="alert(1)" />',
        bidi: '\u202e',
      }}
    />,
  );
  expect(html).not.toContain('<img');
  expect(html).not.toContain('<svg');
  expect(html).toContain('&lt;');
  expect(html).toContain('\\u202e');
  expect(fetcher).not.toHaveBeenCalled();
});
it('schema validation requires no runtime Function/eval capability', () => {
  const original = globalThis.Function;
  const forbidden = vi.fn(() => {
    throw new Error('CSP forbids dynamic Function');
  });
  Object.defineProperty(globalThis, 'Function', { value: forbidden, configurable: true });
  try {
    expect(validateDocument(emptyDocument()).schema).toBe('mk.studio/1');
    expect(z.strictObject({ test: z.string() }).parse({ test: 'ok' })).toEqual({ test: 'ok' });
    expect(forbidden).not.toHaveBeenCalled();
  } finally {
    Object.defineProperty(globalThis, 'Function', { value: original, configurable: true });
  }
});
