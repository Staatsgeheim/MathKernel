// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { webcrypto } from 'node:crypto';
import { WorkflowPanel } from '../src/host/WorkflowPanel';
import { emptyDocument } from '../src/editor/document';
import type { HostCommandService } from '../src/host/client';
import type { Handshake } from '../src/host/contracts';
import type { Frozen } from '../src/host/workflow';
const host: Handshake = {
  host_instance_id: 'h',
  workspace_id: 'w',
  mathkernel_version: 'fixture',
  ui_protocol: 'studio-host/1',
  catalog_revision: 'c',
  test_host: true,
  authenticated: true,
  features: {
    catalog_read: true,
    authoring_draft: true,
    operation_invoke: false,
    workflow_validate: true,
    workflow_execute: true,
    run_observe: true,
    result_inspect: false,
    artifact_view: false,
    compute_review: true,
    approval_interact: true,
  },
  limits: { control_bytes: 2097152, catalog_page: 25 },
  extensions: {
    contract: 'studio-workflow/1',
    scopes: ['workflow_outputs'],
    objects: false,
    subworkflows: false,
  },
};
const key = 'mk-studio-pending:h:w';
beforeEach(() => {
  localStorage.clear();
  Object.defineProperty(globalThis, 'crypto', { value: webcrypto, configurable: true });
  Object.defineProperty(navigator, 'locks', {
    value: {
      request: async (_n: unknown, _o: unknown, fn: (lock: object) => Promise<void>) => fn({}),
    },
    configurable: true,
  });
  HTMLDialogElement.prototype.showModal = function () {
    this.open = true;
  };
  HTMLDialogElement.prototype.close = function () {
    this.open = false;
  };
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});
function setup(lost = false) {
  let time = Date.parse('2026-09-07T00:00:00Z');
  const client = {
    hostNow: () => time,
    validate: vi.fn(async (_d: unknown, binding: Frozen) => ({
      binding,
      validation_ref: 'v',
      valid: true,
      diagnostics: [],
    })),
    plan: vi.fn(async (_v: unknown, binding: Frozen) => ({
      binding,
      plan_ref: 'p',
      digest: 'a'.repeat(64),
      expires_at: '2026-09-07T00:01:00Z',
      scope: 'workflow_outputs',
      selected_nodes: [],
      outputs: [],
      operations: [],
      assumptions: [],
      arithmetic: [],
      evidence_requirements: [],
      resources: [],
      cost: {
        amount: null,
        currency: null,
        source: 'fixture',
        observed_at: '2026-09-07T00:00:00Z',
        uncertainty: 'Unknown',
        exclusions: [],
      },
      exports: [],
      alternatives: [],
      authorization_required: true,
      policy_denied: false,
      warnings: [],
    })),
    challenge: vi.fn(async () => ({
      challenge_ref: 'c',
      plan_ref: 'p',
      plan_digest: 'a'.repeat(64),
      expires_at: '2026-09-07T00:01:00Z',
      disclosures: ['Fixture only'],
      can_confirm: true,
    })),
    confirm: vi.fn(async () => ({
      plan_ref: 'p',
      plan_digest: 'a'.repeat(64),
      authority_ref: 'auth',
      decision: 'approved',
      expires_at: '2026-09-07T00:01:00Z',
    })),
    submit: vi.fn(async (_p: unknown, _d: unknown, _a: unknown, request: string) => {
      if (lost) throw new Error('Acknowledgement lost');
      return {
        client_request_id: request,
        plan_ref: 'p',
        outcome: 'accepted',
        run_ref: 'r',
        message: 'fixture',
      };
    }),
    reconcile: vi.fn(async (request: string) => ({
      client_request_id: request,
      plan_ref: 'p',
      outcome: 'accepted',
      run_ref: 'r',
      message: 'existing request',
    })),
  };
  const props = {
    document: emptyDocument(),
    selection: [],
    host,
    client: client as unknown as HostCommandService,
    onClose: vi.fn(),
    onHostError: vi.fn(),
  };
  return {
    client,
    props,
    setTime: (v: number) => {
      time = v;
    },
  };
}
async function review() {
  fireEvent.click(screen.getByRole('button', { name: 'Validate frozen draft' }));
  await waitFor(() =>
    expect(
      (screen.getByRole('button', { name: 'Request immutable plan' }) as HTMLButtonElement)
        .disabled,
    ).toBe(false),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Request immutable plan' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Request host approval challenge' }));
  fireEvent.click(await screen.findByRole('button', { name: 'Confirm disclosed scope' }));
  await waitFor(() =>
    expect(
      (screen.getByRole('button', { name: 'Submit reviewed plan' }) as HTMLButtonElement).disabled,
    ).toBe(false),
  );
}
it('explicit review and lost acknowledgement reconciliation survive remount without replay', async () => {
  const { props, client } = setup(true);
  const ui = render(<WorkflowPanel {...props} />);
  expect(client.validate).not.toHaveBeenCalled();
  expect(client.submit).not.toHaveBeenCalled();
  await review();
  fireEvent.click(screen.getByRole('button', { name: 'Submit reviewed plan' }));
  await screen.findByText('Acknowledgement lost');
  expect(client.submit).toHaveBeenCalledTimes(1);
  expect(localStorage.getItem(key)).toContain('"plan":"p"');
  expect(localStorage.getItem(key)).not.toContain('auth');
  ui.unmount();
  render(<WorkflowPanel {...props} />);
  fireEvent.click(await screen.findByRole('button', { name: 'Check existing request' }));
  await waitFor(() => expect(localStorage.getItem(key)).toBeNull());
  expect(client.reconcile).toHaveBeenCalledTimes(1);
  expect(client.submit).toHaveBeenCalledTimes(1);
});
it('host expiry disables submission', async () => {
  const { props, setTime } = setup();
  const ui = render(<WorkflowPanel {...props} />);
  await review();
  setTime(Date.parse('2026-09-07T00:02:00Z'));
  ui.rerender(<WorkflowPanel {...props} />);
  expect(
    (screen.getByRole('button', { name: 'Submit reviewed plan' }) as HTMLButtonElement).disabled,
  ).toBe(true);
  expect(screen.getByText('Plan expired. Request a new immutable plan.')).toBeTruthy();
});
it('corrupt pending storage fails closed without commands', async () => {
  const { props, client } = setup();
  localStorage.setItem(key, '{invalid');
  render(<WorkflowPanel {...props} />);
  await screen.findByText(/Command recovery storage is unavailable/);
  expect(client.submit).not.toHaveBeenCalled();
});
