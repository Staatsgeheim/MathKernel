import { useState } from 'react';
import { admittedResult, receipt, verifyPages } from './result';
import { HostError, type HostCommandService } from '../host/client';
import { mathResultSchema, type ResultObservation } from '../host/contracts';
import { JsonView, Text } from '../app/components';
export function ResultInspector({
  observation,
  client,
  onHostError,
}: {
  observation: ResultObservation;
  client: HostCommandService;
  onHostError: (error: unknown) => void;
}) {
  const [content, setContent] = useState('');
  const [offset, setOffset] = useState<number | null>(0);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [complete, setComplete] = useState<unknown>(null);
  const [pageIdentity, setPageIdentity] = useState<{ digest: string; bytes: number } | null>(null);
  const r = receipt(observation.result);
  const accepted = admittedResult(observation);
  const parsed = mathResultSchema.safeParse(complete ?? observation.result);
  const shown = accepted ?? (parsed.success ? parsed.data : null);
  async function nextPage() {
    if (!r || offset === null || busy) return;
    setBusy(true);
    setError('');
    try {
      const page = await client.resultPage(observation.binding.result_ref, r.resource_id, offset);
      if (page.total_bytes > 2097152)
        throw new Error(
          'Result exceeds the 2 MiB inspection budget. Use an existing host export outside Studio.',
        );
      if (
        page.total_bytes !== r.size_bytes ||
        (r.sha256 && r.sha256 !== page.sha256) ||
        (pageIdentity &&
          (pageIdentity.digest !== page.sha256 || pageIdentity.bytes !== page.total_bytes))
      )
        throw new Error('Result resource identity changed.');
      if (
        !/^[\x00-\x7f]*$/.test(page.content) ||
        page.content.length === 0 ||
        (page.next_offset === null && offset + page.content.length !== page.total_bytes)
      )
        throw new Error('Invalid result page boundaries.');
      setPageIdentity({ digest: page.sha256, bytes: page.total_bytes });
      const all = content + page.content;
      if (page.next_offset === null)
        setComplete(await verifyPages(all, page.total_bytes, page.sha256));
      setContent(all);
      setOffset(page.next_offset);
    } catch (e) {
      if (e instanceof HostError && [401, 403].includes(e.status)) onHostError(e);
      else setError(e instanceof Error ? e.message : 'Page read failed.');
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="result-content">
      <header>
        <p className="eyebrow">Read-only observation</p>
        <h2>Result inspection</h2>
        <p className="wrap">
          <Text>{observation.binding.result_ref}</Text>
        </p>
      </header>
      <p className={observation.admission === 'host' ? 'notice' : 'warning'}>
        {observation.admission === 'host'
          ? 'Supplied by the connected host'
          : observation.admission === 'test_fixture'
            ? 'TEST FIXTURE — no mathematics was computed; evidence labels are synthetic.'
            : 'UNTRUSTED CANDIDATE — source labels are not admitted evidence.'}
      </p>
      <dl className="facts">
        <dt>Host / workspace</dt>
        <dd>
          {observation.scope.host_instance_id} / {observation.scope.workspace_id}
        </dd>
        <dt>Source revision</dt>
        <dd>
          {observation.binding.source_ref} / {observation.binding.source_revision}
        </dd>
        <dt>Run / node</dt>
        <dd>
          {observation.binding.run_ref ?? 'No workflow run mapping'} /{' '}
          {observation.binding.node_id ?? 'not supplied'}
        </dd>
        <dt>Frozen draft revision</dt>
        <dd>{observation.binding.draft_revision ?? 'not supplied'}</dd>
        <dt>Observed (UTC)</dt>
        <dd>{observation.observed_at}</dd>
      </dl>
      {r && (
        <section className="receipt">
          <h3>Output-budget receipt</h3>
          <p>
            This is a delivery receipt, not the mathematical result. Original-trust metadata grants
            no proof badge.
          </p>
          <p>
            {content.length} of {r.size_bytes} bytes loaded ·{' '}
            {complete ? 'Integrity checked; source data shown below' : 'Partial result'}
          </p>
          {offset !== null && (
            <button onClick={() => void nextPage()} disabled={busy || r.size_bytes > 2097152}>
              {busy ? 'Reading page…' : 'Load next result page'}
            </button>
          )}
          {complete !== null && (
            <p className="notice">
              Reconstructed source data. Per-claim admission summaries require a full host result
              observation; no new badge is inferred from these bytes.
            </p>
          )}
        </section>
      )}
      {accepted && (
        <div className="result-axes">
          <div>
            <span>Host-admitted trust</span>
            <strong>{accepted.trust}</strong>
          </div>
          <div>
            <span>Semantic status</span>
            <strong>{accepted.semantic_status ?? 'unknown'}</strong>
          </div>
          <div>
            <span>Engine</span>
            <strong>{accepted.engine ?? 'not reported'}</strong>
          </div>
        </div>
      )}
      {shown && (
        <>
          <section>
            <h3>Value / existing artifact data</h3>
            <JsonView value={shown.data} />
            <p className="muted">
              Structured text only. HTML, SVG, scripts and artifact URLs are inert. Rendering does
              not add evidence.
            </p>
          </section>
          <section>
            <h3>Assumptions and arithmetic ancestry</h3>
            <JsonView
              value={{
                assumptions: shown.assumptions_used,
                side_conditions: shown.side_conditions,
                arithmetic_transition: shown.arithmetic_transition,
              }}
            />
          </section>
          <section>
            <h3>{accepted ? 'Per-claim evidence' : 'Source claim metadata (not admitted)'}</h3>
            {Object.entries(shown.claim_evidence).length === 0 && (
              <p>No per-claim evidence supplied.</p>
            )}
            {Object.entries(shown.claim_evidence).map(([claim, bundle]) => (
              <details key={claim} className="claim" open>
                <summary>
                  <Text>{claim}</Text> ·{' '}
                  {accepted
                    ? (observation.claim_trust[claim] ?? 'host claim trust not reported')
                    : 'untrusted / fixture metadata'}
                </summary>
                <p className="muted">
                  Input-ancestry trust ceiling: {bundle.justified_trust ?? 'not reported'}
                </p>
                {(
                  [
                    'computation',
                    'proof',
                    'certificate',
                    'numerical',
                    'model',
                    'empirical',
                  ] as const
                ).map(
                  (group) =>
                    bundle[group].length > 0 && (
                      <section key={group}>
                        <h4>{group}</h4>
                        {bundle[group].map((record, i) => (
                          <div key={i} className="evidence-record">
                            <p>
                              Role: <Text>{String(record.role ?? 'not reported')}</Text> · Support
                              path: <Text>{String(record.support_path ?? 'not reported')}</Text>
                            </p>
                            <JsonView value={record} />
                          </div>
                        ))}
                      </section>
                    ),
                )}
              </details>
            ))}
          </section>
          <section>
            <h3>Warnings and errors</h3>
            <JsonView value={{ warnings: shown.warnings, errors: shown.errors }} />
          </section>
          <details>
            <summary>Derivation and source provenance</summary>
            <JsonView
              value={{
                derivation: shown.derivation,
                engine_versions: shown.engine_versions,
                mathkernel_version: shown.mathkernel_version,
              }}
            />
          </details>
        </>
      )}
      {!shown && !r && (
        <>
          <h3>Inert source data</h3>
          <JsonView value={observation.result} />
        </>
      )}
      {error && (
        <p role="alert" className="warning">
          {error}
        </p>
      )}
    </div>
  );
}
