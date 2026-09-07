import { id } from '../editor/document';
import type { Submission } from './workflow';

export type Pending = { request: string; plan: string | null };
export function readPending(storage: Pick<Storage, 'getItem'>, key: string): Pending | null {
  const raw = storage.getItem(key);
  if (raw === null) return null;
  if (raw.length > 512) throw new Error('Pending command journal exceeds its budget.');
  // Backward compatibility with the 0.2 correlation-only journal.
  if (id.safeParse(raw).success) return { request: raw, plan: null };
  const value = JSON.parse(raw);
  if (!value || Object.keys(value).sort().join(',') !== 'plan,request')
    throw new Error('Invalid pending command journal.');
  return {
    request: id.parse(value.request),
    plan: value.plan === null ? null : id.parse(value.plan),
  };
}
export function retainPending(
  storage: Pick<Storage, 'getItem' | 'setItem'>,
  key: string,
  value: Pending,
) {
  if (readPending(storage, key))
    throw new Error('An unresolved command already exists for this host. Reconcile it first.');
  const raw = JSON.stringify(value);
  storage.setItem(key, raw);
  if (storage.getItem(key) !== raw)
    throw new Error('Could not retain command correlation. Nothing was submitted.');
}
export function checkReply(reply: Submission, pending: Pending) {
  if (
    reply.client_request_id !== pending.request ||
    (pending.plan && reply.plan_ref !== pending.plan) ||
    (reply.outcome === 'accepted' && !reply.run_ref)
  )
    throw new Error('Command identity or acknowledgement mismatch. Outcome remains unresolved.');
}
