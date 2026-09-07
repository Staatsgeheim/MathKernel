import { diagnostics } from '../editor/commands';
import { validateDocument } from '../editor/document';
import { operationSchema } from '../host/contracts';
self.onmessage = (event: MessageEvent) => {
  const { request, document, catalog } = event.data;
  try {
    self.postMessage({
      request,
      problems: diagnostics(
        validateDocument(document),
        operationSchema.array().max(20000).parse(catalog),
      ),
    });
  } catch {
    self.postMessage({ request, error: 'Diagnostic input exceeds the supported contract.' });
  }
};
