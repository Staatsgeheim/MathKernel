import { importDocument, serializeDocument, type StudioDocument } from './document';
const DB = 'mathkernel-studio-recovery-v1';
export type RecoveryEntry = { key: string; version: number; text: string; updatedAt: string };
export function recoveryScope(hostId: string, workspaceId: string, documentId: string): string {
  return JSON.stringify([hostId, workspaceId, documentId]);
}
export async function listRecovery(hostId: string, workspaceId: string): Promise<RecoveryEntry[]> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('drafts', 'readonly');
    const r = tx.objectStore('drafts').getAll(undefined, 10);
    tx.oncomplete = () => {
      db.close();
      resolve(
        (r.result as RecoveryEntry[])
          .filter((e) => {
            try {
              const s = JSON.parse(e.key);
              return s[0] === hostId && s[1] === workspaceId;
            } catch {
              return false;
            }
          })
          .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
      );
    };
    tx.onabort = tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}
function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB, 1);
    r.onupgradeneeded = () => r.result.createObjectStore('drafts', { keyPath: 'key' });
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error ?? new Error('Recovery storage unavailable.'));
    r.onblocked = () => reject(new Error('Recovery storage upgrade blocked by another tab.'));
  });
}
export async function loadRecovery(key: string): Promise<RecoveryEntry | null> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('drafts', 'readonly');
    const r = tx.objectStore('drafts').get(key);
    tx.oncomplete = () => {
      db.close();
      resolve(r.result ?? null);
    };
    tx.onabort = tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}
export async function saveRecovery(
  key: string,
  document: StudioDocument,
  expectedVersion: number,
): Promise<number> {
  const text = serializeDocument(document);
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('drafts', 'readwrite');
    const store = tx.objectStore('drafts');
    let conflict = false;
    const read = store.get(key);
    read.onsuccess = () => {
      if ((read.result?.version ?? 0) !== expectedVersion) {
        conflict = true;
        tx.abort();
        return;
      }
      const count = store.count();
      count.onsuccess = () => {
        if (!read.result && count.result >= 10) {
          tx.abort();
          return;
        }
        store.put({ key, version: expectedVersion + 1, text, updatedAt: new Date().toISOString() });
      };
    };
    tx.oncomplete = () => {
      db.close();
      resolve(expectedVersion + 1);
    };
    tx.onabort = tx.onerror = () => {
      db.close();
      reject(
        conflict
          ? new Error(
              'Another tab changed this recovery copy. Export your draft before resolving the conflict.',
            )
          : (tx.error ?? new Error('Browser storage quota or policy prevented recovery.')),
      );
    };
  });
}
export async function deleteRecovery(key: string, expectedVersion: number): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('drafts', 'readwrite');
    const store = tx.objectStore('drafts');
    let conflict = false;
    const r = store.get(key);
    r.onsuccess = () => {
      if ((r.result?.version ?? 0) !== expectedVersion) {
        conflict = true;
        tx.abort();
      } else store.delete(key);
    };
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onabort = tx.onerror = () => {
      db.close();
      reject(
        new Error(
          conflict ? 'Recovery copy changed in another tab.' : 'Could not remove recovery copy.',
        ),
      );
    };
  });
}
export function restoreRecovery(entry: RecoveryEntry): StudioDocument {
  return importDocument(entry.text);
}
