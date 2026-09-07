import { randomId } from '../security/identity';
import type { StudioDocument } from './document';
import { DOCUMENT_BYTES } from '../security/json';
export function readDocument(file: File, signal?: AbortSignal): Promise<StudioDocument> {
  if (file.size > DOCUMENT_BYTES) return Promise.reject(new Error('Document exceeds 10 MiB.'));
  if (!file.name.endsWith('.mkstudio.json'))
    return Promise.reject(new Error('Select a .mkstudio.json document.'));
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('../workers/import.worker.ts', import.meta.url), {
      type: 'module',
    });
    const requestId = randomId();
    const cleanup = () => {
      clearTimeout(timer);
      worker.terminate();
      signal?.removeEventListener('abort', cancel);
    };
    const cancel = () => {
      cleanup();
      reject(new Error('Import cancelled.'));
    };
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error('Import exceeded the 5-second parsing budget.'));
    }, 5000);
    signal?.addEventListener('abort', cancel, { once: true });
    if (signal?.aborted) {
      cancel();
      return;
    }
    worker.onmessage = (e) => {
      if (e.data.requestId !== requestId) return;
      cleanup();
      e.data.error ? reject(new Error(e.data.error)) : resolve(e.data.document);
    };
    worker.onerror = () => {
      cleanup();
      reject(new Error('Import worker failed; current draft preserved.'));
    };
    file
      .text()
      .then((text) => {
        if (!signal?.aborted) worker.postMessage({ requestId, text });
      })
      .catch((error) => {
        cleanup();
        reject(error);
      });
  });
}
