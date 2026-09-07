import { importDocument } from '../editor/document';
self.onmessage = (event: MessageEvent<{ requestId: string; text: string }>) => {
  try {
    self.postMessage({
      requestId: event.data.requestId,
      document: importDocument(event.data.text),
    });
  } catch (error) {
    self.postMessage({
      requestId: event.data.requestId,
      error: error instanceof Error ? error.message.slice(0, 1000) : 'Import failed.',
    });
  }
};
