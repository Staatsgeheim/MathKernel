import 'fake-indexeddb/auto';
import { expect, it } from 'vitest';
import {
  deleteRecovery,
  listRecovery,
  loadRecovery,
  recoveryScope,
  restoreRecovery,
  saveRecovery,
} from '../src/editor/recovery';
import { emptyDocument, makeInput } from '../src/editor/document';
it('uses host/workspace/document keys and preserves invalid exact drafts', async () => {
  const d = emptyDocument();
  d.authoring.nodes.push(makeInput('integer'));
  d.authoring.nodes[0]!.parameter_drafts.value!.text = '9007199254740993x';
  const key = recoveryScope('host', 'workspace', d.identity.document_id);
  expect(await saveRecovery(key, d, 0)).toBe(1);
  const r = await loadRecovery(key);
  expect(restoreRecovery(r!).authoring.nodes[0]!.parameter_drafts.value!.text).toBe(
    '9007199254740993x',
  );
  expect(await listRecovery('other-host', 'workspace')).toEqual([]);
  expect(
    await loadRecovery(recoveryScope('host', 'other-workspace', d.identity.document_id)),
  ).toBeNull();
});
it('rejects conflicting tabs without overwriting either caller draft', async () => {
  const d = emptyDocument(),
    key = recoveryScope('host', 'workspace', d.identity.document_id);
  await saveRecovery(key, d, 0);
  const outcomes = await Promise.allSettled([saveRecovery(key, d, 1), saveRecovery(key, d, 1)]);
  expect(outcomes.filter((r) => r.status === 'fulfilled')).toHaveLength(1);
  expect(outcomes.filter((r) => r.status === 'rejected')).toHaveLength(1);
  expect((await loadRecovery(key))!.version).toBe(2);
  await expect(deleteRecovery(key, 1)).rejects.toThrow('changed');
  await deleteRecovery(key, 2);
  expect(await loadRecovery(key)).toBeNull();
});
it('surfaces browser storage policy/quota failure', async () => {
  const original = globalThis.indexedDB;
  Object.defineProperty(globalThis, 'indexedDB', {
    value: {
      open() {
        throw new Error('Quota/policy denied');
      },
    },
    configurable: true,
  });
  try {
    await expect(saveRecovery('x', emptyDocument(), 0)).rejects.toThrow('Quota/policy');
  } finally {
    Object.defineProperty(globalThis, 'indexedDB', { value: original, configurable: true });
  }
});
