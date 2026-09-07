import { useEffect, useRef, useState } from 'react';
import type { StudioDocument } from './document';
import type { Diagnostic } from './commands';
import type { Operation } from '../host/contracts';

export function useDiagnostics(document: StudioDocument, catalog: Operation[]) {
  const [problems, setProblems] = useState<Diagnostic[]>([]);
  const generation = useRef(0);
  const key = `${document.identity.document_id}:${document.authoring.revision}`;
  useEffect(() => {
    const request = ++generation.current;
    let worker: Worker | undefined;
    const timer = setTimeout(() => {
      try {
        worker = new Worker(new URL('../workers/diagnostics.worker.ts', import.meta.url), {
          type: 'module',
        });
        worker.onmessage = (event) => {
          if (event.data.request === generation.current)
            setProblems(
              event.data.problems ?? [
                {
                  code: 'CHECK_UNAVAILABLE',
                  message: event.data.error,
                  revision: document.authoring.revision,
                  layer: 'editor',
                },
              ],
            );
          worker?.terminate();
          worker = undefined;
        };
        worker.onerror = () => {
          setProblems([
            {
              code: 'CHECK_UNAVAILABLE',
              message: 'Worker check unavailable. Host validation remains required.',
              revision: document.authoring.revision,
              layer: 'editor',
            },
          ]);
          worker?.terminate();
          worker = undefined;
        };
        worker.postMessage({ request, document, catalog });
      } catch {
        setProblems([
          {
            code: 'CHECK_UNAVAILABLE',
            message: 'Worker checks unavailable.',
            revision: document.authoring.revision,
            layer: 'editor',
          },
        ]);
      }
    }, 200);
    const deadline = setTimeout(() => {
      if (worker) {
        worker.terminate();
        setProblems([
          {
            code: 'CHECK_TIMEOUT',
            message:
              'Advisory check exceeded five seconds. Draft retained; host validation required.',
            revision: document.authoring.revision,
            layer: 'editor',
          },
        ]);
      }
    }, 5000);
    return () => {
      generation.current++;
      clearTimeout(timer);
      clearTimeout(deadline);
      worker?.terminate();
    };
  }, [key, catalog]);
  return problems;
}
