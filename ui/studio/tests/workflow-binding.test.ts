import { expect, it } from 'vitest';
import vector from '../qa/workflow-binding.json';
import { validateDocument } from '../src/editor/document';
import { freezeDocument } from '../src/host/workflow';
it('Python and TypeScript agree on the exact frozen executable document digest', async () => {
  expect(await freezeDocument(validateDocument(vector.document))).toEqual(vector.binding);
});
