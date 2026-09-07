import { randomId } from '../security/identity';
import { useRef, useState } from 'react';
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
}: {
  document: StudioDocument;
  selection: string[];
  host: Handshake;
  client: HostCommandService;
  onClose: () => void;
  onHostError: (e: unknown) => void;
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
  const [pending, setPending] = useState<string | null>(() => {
    try {
      const value = localStorage.getItem(storageKey);
      return value ? id.parse(value) : null;
    } catch {
      return null;
    }
  });
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
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Host command failed.');
      setFreshness('Stale / request failed');
      onHostError(e);
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }
  function retain(request: string) {
    // Persist correlation only. Never persist authority or a command for replay.
    localStorage.setItem(storageKey, request);
    setPending(request);
  }
  function resolve(reply: Submission, expected: string) {
    if (reply.client_request_id !== expected)
      throw new Error('Command correlation mismatch. Outcome remains unknown.');
    setSubmission(reply);
    if (reply.outcome !== 'unknown') {
      localStorage.removeItem(storageKey);
      setPending(null);
    }
  }
  async function observe(ref: string) {
    const snapshot = await client.run(id.parse(ref));
    setRun(run?.run_ref === ref ? acceptSnapshot(run, snapshot) : snapshot);
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
            !!pending
          }
          onClick={() =>
            void act(async () => {
              if (!validation || !sameBinding(validation.binding, await freezeDocument(document)))
                throw new Error('Draft changed; validate again.');
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
            Plan {plan.plan_ref} · expires {plan.expires_at}
          </p>
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
          <h4>Resources and cost</h4>
          <p>
            {plan.cost.amount === null
              ? 'Cost unknown'
              : `${plan.cost.amount} ${plan.cost.currency ?? '(currency not reported)'}`}{' '}
            · {plan.cost.uncertainty}
          </p>
          <JsonView
            value={{ resources: plan.resources, cost: plan.cost, alternatives: plan.alternatives }}
          />
          <h4>Data export closure</h4>
          <p>
            These are the host-resolved inputs and destinations, including any ancestors it reports.
          </p>
          <JsonView value={plan.exports} />
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
                Date.parse(plan.expires_at) <= Date.now()
              }
              onClick={() =>
                void act(async () => {
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
                  disabled={
                    busy ||
                    !current(plan) ||
                    plan.policy_denied ||
                    !challenge.can_confirm ||
                    Date.parse(challenge.expires_at) <= Date.now()
                  }
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') e.preventDefault();
                  }}
                  onClick={() =>
                    void act(async () => {
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
              submission?.outcome === 'accepted' ||
              !current(plan) ||
              plan.policy_denied ||
              Date.parse(plan.expires_at) <= Date.now() ||
              (plan.authorization_required &&
                (!approval?.authority_ref ||
                  approval.decision !== 'approved' ||
                  Date.parse(approval.expires_at) <= Date.now()))
            }
            onClick={() =>
              void act(async () => {
                if (
                  !sameBinding(plan.binding, await freezeDocument(document)) ||
                  Date.parse(plan.expires_at) <= Date.now()
                )
                  throw new Error('Draft changed or plan expired.');
                const request = randomId();
                retain(request);
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
                resolve(reply, request);
              })
            }
          >
            Submit reviewed plan
          </button>
        </section>
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
                setRuns((old) => [...old, ...page.runs].slice(-1000));
                setRunOffset(page.next_offset);
              })
            }
          >
            Load run references
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
          <JsonView value={run.attempts} label="Node attempt history" />
          <JsonView value={run.events} label="Recent run events; at most 500" />
          {run.actions.map((action) => (
            <button
              key={action}
              disabled={busy || !!pending}
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
                disabled={busy || !!pending}
                onClick={() =>
                  void act(async () => {
                    const request = randomId();
                    retain(request);
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
