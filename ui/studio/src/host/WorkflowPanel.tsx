import { ComputeReview } from './ComputeReview';
import { readPending, retainPending, checkReply, type Pending } from './pending';
import { randomId } from '../security/identity';
import { useEffect, useRef, useState } from 'react';
import type { Handshake } from './contracts';
import type { HostCommandService } from './client';
import { Dialog, JsonView, Text } from '../app/components';
import { id, type StudioDocument } from '../editor/document';
import {
  freezeDocument,
  sameBinding,
  acceptSnapshot,
  type Plan,
  type Run,
  type Submission,
  validationSchema,
  challengeSchema,
  approvalSchema,
} from './workflow';
import type { z } from '../security/schema';

type Validation = z.infer<typeof validationSchema>;
type Challenge = z.infer<typeof challengeSchema>;
type Approval = z.infer<typeof approvalSchema>;
export function WorkflowPanel({
  document,
  selection,
  host,
  client,
  onClose,
  onHostError,
  onResult,
}: {
  document: StudioDocument;
  selection: string[];
  host: Handshake;
  client: HostCommandService;
  onClose: () => void;
  onHostError: (e: unknown) => void;
  onResult?: (result: import('./contracts').ResultObservation) => void;
}) {
  const [validation, setValidation] = useState<Validation | null>(null),
    [plan, setPlan] = useState<Plan | null>(null);
  const [challenge, setChallenge] = useState<Challenge | null>(null),
    [approval, setApproval] = useState<Approval | null>(null);
  const [submission, setSubmission] = useState<Submission | null>(null),
    [run, setRun] = useState<Run | null>(null);
  const [runs, setRuns] = useState<
      { run_ref: string; draft_revision: number; execution: string }[]
    >([]),
    [runOffset, setRunOffset] = useState<number | null>(0);
  const [busy, setBusy] = useState(false),
    lock = useRef(false),
    [error, setError] = useState(''),
    [freshness, setFreshness] = useState('Not observed');
  const [scope, setScope] = useState(host.extensions?.scopes[0] ?? 'workflow_outputs');
  const [reference, setReference] = useState('');
  const storageKey = `mk-studio-pending:${host.host_instance_id}:${host.workspace_id}`;
  const [journalError, setJournalError] = useState('');
  const [pendingRecord, setPendingRecord] = useState<Pending | null>(null);
  useEffect(() => {
    const refresh = () => {
      try {
        setPendingRecord(readPending(localStorage, storageKey));
        setJournalError('');
      } catch {
        setJournalError(
          'Command recovery storage is unavailable or corrupt. Submission is blocked; preserve browser storage and reconcile with the host operator.',
        );
      }
    };
    refresh();
    window.addEventListener('storage', refresh);
    return () => window.removeEventListener('storage', refresh);
  }, [storageKey]);
  const pending = pendingRecord?.request ?? null;
  // Tick display gates so a quiet, open approval cannot stay enabled past expiry.
  const [, tick] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const latestDocument = useRef(document);
  latestDocument.current = document;
  const [actionReview, setActionReview] = useState<Run['actions'][number] | null>(null);
  const compatible = host.extensions?.contract === 'studio-workflow/1';
  const current = (p: Plan | Validation) =>
    p.binding.document_id === document.identity.document_id &&
    p.binding.draft_revision === document.authoring.revision;
  async function act(action: () => Promise<void>) {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setError('');
    try {
      if (navigator.locks)
        await navigator.locks.request(
          `studio-command:${storageKey}`,
          { ifAvailable: true },
          async (lease) => {
            if (!lease)
              throw new Error(
                'Another tab is reviewing a host command. Wait and reconcile its outcome.',
              );
            await action();
          },
        );
      else await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Host command failed.');
      setFreshness('Stale / request failed');
      onHostError(e);
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }
  function retain(request: string, planRef: string) {
    if (!navigator.locks)
      throw new Error(
        'This browser cannot coordinate commands across tabs. Submission is unavailable.',
      );
    const value = { request, plan: planRef };
    retainPending(localStorage, storageKey, value);
    setPendingRecord(value);
  }
  function resolve(reply: Submission, expected: string, planRef?: string) {
    const record = readPending(localStorage, storageKey);
    if (!record || record.request !== expected)
      throw new Error('Pending command journal changed. Reconcile before another action.');
    checkReply(reply, { request: expected, plan: planRef ?? record.plan });
    setSubmission(reply);
    if (reply.outcome !== 'unknown') {
      localStorage.removeItem(storageKey);
      setPendingRecord(null);
    }
  }
  async function observe(ref: string) {
    const snapshot = await client.run(id.parse(ref));
    setRun(acceptSnapshot(run?.run_ref === ref ? run : null, snapshot));
    setActionReview(null);
    setFreshness('Current as of the last explicit refresh');
  }
  return (
    <Dialog title="Workflow plans and run observations" onClose={onClose}>
      {host.test_host && (
        <p className="warning">
          TEST HOST — every plan, approval and run below is synthetic. No mathematics, provisioning
          or spending occurs.
        </p>
      )}
      {!compatible && (
        <p className="notice">
          This host has no compatible workflow service. Authoring and result inspection remain
          available.
        </p>
      )}
      <p>
        Current draft {document.authoring.revision} · {document.identity.document_id}
      </p>
      <div className="toolbar">
        <button
          disabled={busy || !compatible || !host.features.workflow_validate}
          onClick={() =>
            void act(async () => {
              const snapshot = structuredClone(document),
                binding = await freezeDocument(snapshot);
              const value = await client.validate(snapshot, binding);
              if (!sameBinding(binding, value.binding))
                throw new Error('Validation is bound to a different draft.');
              setValidation(value);
              setPlan(null);
              setChallenge(null);
              setApproval(null);
            })
          }
        >
          Validate frozen draft
        </button>
        <label className="field">
          Requested scope
          <select
            value={scope}
            onChange={(e) => {
              setScope(e.target.value as typeof scope);
              setPlan(null);
              setChallenge(null);
              setApproval(null);
            }}
          >
            {(host.extensions?.scopes ?? []).map((s) => (
              <option key={s} value={s}>
                {s.replaceAll('_', ' ')}
              </option>
            ))}
          </select>
        </label>
        <button
          disabled={
            busy ||
            !validation?.valid ||
            !current(validation) ||
            !host.features.workflow_execute ||
            !!pending ||
            !!journalError
          }
          onClick={() =>
            void act(async () => {
              if (
                !validation ||
                !sameBinding(validation.binding, await freezeDocument(latestDocument.current))
              )
                throw new Error('Draft changed; validate again.');
              if (
                scope !== 'workflow_outputs' &&
                (!selection.length || (scope === 'standalone_operation' && selection.length !== 1))
              )
                throw new Error(
                  'Select nodes for this scope; a standalone operation requires exactly one.',
                );
              const value = await client.plan(
                validation.validation_ref,
                validation.binding,
                scope,
                scope === 'workflow_outputs' ? [] : selection,
              );
              if (!sameBinding(value.binding, validation.binding) || value.scope !== scope)
                throw new Error('Plan binding/scope differs from the request.');
              setPlan(value);
              setChallenge(null);
              setApproval(null);
              setSubmission(null);
            })
          }
        >
          Request immutable plan
        </button>
      </div>
      {validation && (
        <section>
          <h3>Host validation · draft {validation.binding.draft_revision}</h3>
          <p>
            {!current(validation)
              ? 'Stale for the current draft'
              : validation.valid
                ? 'Host accepted this draft for planning'
                : 'Host validation did not accept this draft'}
          </p>
          <JsonView value={validation.diagnostics} label="Host diagnostics" />
        </section>
      )}
      {plan && (
        <section>
          <h3>Review plan</h3>
          <p>
            Plan {plan.plan_ref} · expires {plan.expires_at} (UTC; display uses approximate host
            time)
          </p>
          {Date.parse(plan.expires_at) <= client.hostNow() && (
            <p className="warning" role="status">
              Plan expired. Request a new immutable plan.
            </p>
          )}
          <p className="wrap">Digest {plan.digest}</p>
          {!current(plan) && (
            <p className="warning">
              This plan belongs to draft {plan.binding.draft_revision}. Revalidate the current draft
              before submitting.
            </p>
          )}
          <JsonView
            value={{
              frozen_binding: plan.binding,
              scope: plan.scope,
              selected_nodes: plan.selected_nodes,
              outputs: plan.outputs,
              resolved_operations: plan.operations,
              assumptions: plan.assumptions,
              arithmetic: plan.arithmetic,
              evidence_requirements: plan.evidence_requirements,
            }}
            label="Frozen plan scope"
          />
          <ComputeReview plan={plan} />
          <JsonView value={plan.warnings} label="Plan warnings" />
          {plan.policy_denied && (
            <p className="warning">
              Host policy denied this plan. Confirmation cannot override that denial.
            </p>
          )}
          {plan.authorization_required && !approval && (
            <button
              disabled={
                busy ||
                plan.policy_denied ||
                !current(plan) ||
                !host.features.approval_interact ||
                Date.parse(plan.expires_at) <= client.hostNow()
              }
              onClick={() =>
                void act(async () => {
                  if (!sameBinding(plan.binding, await freezeDocument(latestDocument.current)))
                    throw new Error('Draft changed; validate again.');
                  const c = await client.challenge(plan.plan_ref, plan.digest);
                  if (c.plan_ref !== plan.plan_ref || c.plan_digest !== plan.digest)
                    throw new Error('Approval challenge is bound to a different plan.');
                  setChallenge(c);
                })
              }
            >
              Request host approval challenge
            </button>
          )}
          {challenge && !approval && (
            <section className="approval-chrome">
              <h4>Host confirmation</h4>
              <p>
                Challenge {challenge.challenge_ref} · expires {challenge.expires_at}
              </p>
              <JsonView value={challenge.disclosures} />
              <p>
                This decision is sent to the host authorization service for this exact plan. It does
                not change host policy.
              </p>
              {(['approve', 'deny'] as const).map((decision) => (
                <button
                  key={decision}
                  onKeyDown={(event) => {
                    if (event.repeat) event.preventDefault();
                  }}
                  disabled={
                    busy ||
                    !current(plan) ||
                    plan.policy_denied ||
                    (decision === 'approve' && !challenge.can_confirm) ||
                    Date.parse(plan.expires_at) <= client.hostNow() ||
                    Date.parse(challenge.expires_at) <= client.hostNow()
                  }
                  onClick={() =>
                    void act(async () => {
                      if (
                        !sameBinding(plan.binding, await freezeDocument(latestDocument.current)) ||
                        Date.parse(challenge.expires_at) <= client.hostNow() ||
                        Date.parse(plan.expires_at) <= client.hostNow()
                      )
                        throw new Error('Draft changed or approval scope expired.');
                      const value = await client.confirm(
                        challenge.challenge_ref,
                        plan.plan_ref,
                        plan.digest,
                        decision,
                      );
                      if (value.plan_ref !== plan.plan_ref || value.plan_digest !== plan.digest)
                        throw new Error('Approval receipt is bound to a different plan.');
                      setApproval(value);
                    })
                  }
                >
                  {decision === 'approve' ? 'Confirm disclosed scope' : 'Deny this plan'}
                </button>
              ))}
            </section>
          )}
          {approval && (
            <p>
              Host decision: {approval.decision} · expires {approval.expires_at}
            </p>
          )}
          <button
            className="primary"
            disabled={
              busy ||
              !!pending ||
              !!journalError ||
              submission?.outcome === 'accepted' ||
              !current(plan) ||
              plan.policy_denied ||
              Date.parse(plan.expires_at) <= client.hostNow() ||
              (plan.authorization_required &&
                (!approval?.authority_ref ||
                  approval.decision !== 'approved' ||
                  Date.parse(approval.expires_at) <= client.hostNow()))
            }
            onClick={() =>
              void act(async () => {
                if (
                  !sameBinding(plan.binding, await freezeDocument(latestDocument.current)) ||
                  Date.parse(plan.expires_at) <= client.hostNow()
                )
                  throw new Error('Draft changed or plan expired.');
                const request = randomId();
                retain(request, plan.plan_ref);
                setSubmission({
                  client_request_id: request,
                  plan_ref: plan.plan_ref,
                  outcome: 'unknown',
                  run_ref: null,
                  message: 'Submission outcome unknown until acknowledged. Do not resubmit.',
                });
                const reply = await client.submit(
                  plan.plan_ref,
                  plan.digest,
                  approval?.authority_ref ?? null,
                  request,
                );
                if (reply.plan_ref !== plan.plan_ref)
                  throw new Error('Submission plan mismatch. Outcome remains unknown.');
                resolve(reply, request, plan.plan_ref);
              })
            }
          >
            Submit reviewed plan
          </button>
        </section>
      )}
      {journalError && (
        <p role="alert" className="warning">
          {journalError}
        </p>
      )}
      {pending && (
        <section className="warning">
          <h3>Command outcome unresolved</h3>
          <p>
            Correlation: {pending}. The host may have accepted the command. No automatic retry will
            occur.
          </p>
          <button
            disabled={busy || !host.features.run_observe}
            onClick={() => void act(async () => resolve(await client.reconcile(pending), pending))}
          >
            Check existing request
          </button>
        </section>
      )}
      {submission && (
        <section>
          <h3>Submission observation</h3>
          <p>
            {submission.outcome} · <Text>{submission.message}</Text>
          </p>
          {submission.run_ref && (
            <button disabled={busy} onClick={() => void act(() => observe(submission.run_ref!))}>
              Inspect submitted run
            </button>
          )}
        </section>
      )}
      <section>
        <h3>Recorded runs</h3>
        <div className="toolbar">
          <button
            disabled={busy || !host.features.run_observe || !compatible || runOffset === null}
            onClick={() =>
              void act(async () => {
                const page = await client.runs(runOffset ?? 0);
                if (page.next_offset !== null && page.next_offset <= (runOffset ?? 0))
                  throw new Error('Run paging did not advance.');
                setRuns((old) =>
                  [...new Map([...old, ...page.runs].map((r) => [r.run_ref, r])).values()].slice(
                    -1000,
                  ),
                );
                setRunOffset(page.next_offset);
              })
            }
          >
            Load run references
          </button>
          <button
            disabled={busy || !host.features.run_observe}
            onClick={() => {
              setRuns([]);
              setRunOffset(0);
            }}
          >
            Restart run listing
          </button>
          <label className="field">
            Run reference
            <input
              value={reference}
              onChange={(e) => setReference(e.target.value)}
              maxLength={128}
            />
          </label>
          <button
            disabled={busy || !host.features.run_observe || !reference}
            onClick={() => void act(() => observe(reference))}
          >
            Open run
          </button>
        </div>
        {runs.map((r) => (
          <button
            key={r.run_ref}
            disabled={busy}
            onClick={() => void act(() => observe(r.run_ref))}
          >
            {r.run_ref} · draft {r.draft_revision} · {r.execution}
          </button>
        ))}
      </section>
      {run && (
        <section>
          <h3>
            Run {run.run_ref} · frozen draft {run.binding.draft_revision}
          </h3>
          <p>
            {freshness} · observed {run.observed_at}
          </p>
          <p>
            {run.binding.document_id !== document.identity.document_id ||
            run.binding.draft_revision !== document.authoring.revision
              ? 'This observation belongs to a different draft. It does not update current node badges.'
              : 'Frozen revision matches the displayed draft revision.'}
          </p>
          <div className="result-axes">
            {(['execution', 'verification', 'artifacts', 'resources', 'cost'] as const).map(
              (axis) => (
                <div key={axis}>
                  <span>{axis}</span>
                  <strong>
                    <Text>{run[axis]}</Text>
                  </strong>
                </div>
              ),
            )}
          </div>
          {run.warnings.map((w, i) => (
            <p key={i} className="warning">
              <Text>{w}</Text>
            </p>
          ))}
          <button disabled={busy} onClick={() => void act(() => observe(run.run_ref))}>
            Refresh run snapshot
          </button>
          <p className="muted">
            Manual snapshots; no background polling. Closing this view does not stop the host run.
          </p>
          <table>
            <caption>Node attempt history</caption>
            <thead>
              <tr>
                <th>Node / attempt</th>
                <th>State</th>
                <th>Progress</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {run.attempts.map((attempt) => (
                <tr key={attempt.attempt_id}>
                  <td>
                    <Text>{attempt.node_id}</Text>
                    <br />
                    <Text>{attempt.attempt_id}</Text>
                  </td>
                  <td>
                    <Text>{attempt.state}</Text>
                  </td>
                  <td>
                    {attempt.progress === null
                      ? 'Indeterminate'
                      : `${attempt.progress}% (${attempt.progress_kind})`}
                  </td>
                  <td>
                    {attempt.result_ref ? (
                      <button
                        disabled={busy || !host.features.result_inspect || !onResult}
                        onClick={() =>
                          void act(async () => {
                            const result = await client.runResult(run, attempt);
                            onResult?.(result);
                          })
                        }
                      >
                        Inspect {attempt.result_ref}
                      </button>
                    ) : (
                      'No result reported'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <JsonView value={run.events} label="Recent run events; at most 500" />
          {run.actions.map((action) => (
            <button
              key={action}
              disabled={busy || !!pending || !!journalError}
              onClick={() => setActionReview(action)}
            >
              Review {action} request
            </button>
          ))}
          {actionReview && (
            <div className="warning">
              <p>
                Request {actionReview} for {run.run_ref} at host revision {run.revision}? The host
                rechecks policy. Cancellation does not imply cleanup or cost settlement.
              </p>
              <button
                disabled={busy || !!pending || !!journalError}
                onClick={() =>
                  void act(async () => {
                    const request = randomId();
                    retain(request, run.plan_ref);
                    resolve(
                      await client.runAction(run.run_ref, run.revision, actionReview, request),
                      request,
                    );
                    setActionReview(null);
                  })
                }
              >
                Send reviewed {actionReview} request
              </button>
              <button onClick={() => setActionReview(null)}>Keep observing</button>
            </div>
          )}
        </section>
      )}
      {error && (
        <p className="warning" role="alert">
          {error}
        </p>
      )}
    </Dialog>
  );
}
