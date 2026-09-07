import type { Operation } from '../host/contracts';
import { emptyDocument, newId, type DraftNode } from './document';

/** Authoring convenience only. It cannot validate, approve or execute the graph. */
export function exactExample(catalog: Operation[]) {
  const doc = emptyDocument();
  doc.identity.title = 'Exact expression → derivative';
  const make = (method: string, params: Record<string, string>): DraftNode => {
    const op = catalog.find(
      (o) => o.operation_ref === `workflow.${method}` && o.composition === 'complete',
    );
    if (!op) throw new Error('This host does not advertise the local expression adapters.');
    return {
      id: newId(),
      kind: 'operation',
      label: op.title,
      operation_ref: op.operation_ref,
      operation_version: op.operation_version,
      schema_digest: op.schema_digest,
      parameter_drafts: Object.fromEntries(
        Object.entries(params).map(([k, text]) => [k, { encoding: 'text', text }]),
      ),
      input_bindings: {},
    };
  };
  const input = make('parse', { expression: 'x^3 + 9007199254740993' }),
    derivative = make('differentiate', { variable: 'x' });
  doc.authoring.nodes = [input, derivative];
  doc.authoring.edges = [
    {
      id: newId(),
      source_node: input.id,
      source_port: 'value',
      target_node: derivative.id,
      target_port: 'expr_id',
    },
  ];
  doc.authoring.desired_outputs = [{ node_id: derivative.id, port_id: 'value' }];
  doc.authoring.revision = 1;
  doc.presentation.node_positions = {
    [input.id]: { x: 60, y: 60 },
    [derivative.id]: { x: 410, y: 60 },
  };
  return doc;
}
