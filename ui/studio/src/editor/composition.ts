import { emptyDocument, validateDocument, type StudioDocument } from './document';

/** A fragment is a copy of authoring intent. Boundary edges are disclosed, never recreated. */
export function selectionFragment(document: StudioDocument, selection: string[], title: string) {
  const ids = new Set(selection);
  if (!ids.size) throw new Error('Select at least one node.');
  const fragment = emptyDocument();
  fragment.identity.title = title;
  fragment.authoring.nodes = structuredClone(document.authoring.nodes.filter((n) => ids.has(n.id)));
  const bindings = fragment.authoring.nodes.flatMap((n) =>
    Object.keys(n.input_bindings).map((port) => ({ node: n.id, port })),
  );
  fragment.authoring.nodes.forEach((n) => {
    n.input_bindings = {};
  });
  fragment.authoring.edges = structuredClone(
    document.authoring.edges.filter((e) => ids.has(e.source_node) && ids.has(e.target_node)),
  );
  fragment.authoring.desired_outputs = structuredClone(
    document.authoring.desired_outputs.filter((o) => ids.has(o.node_id)),
  );
  fragment.presentation.node_positions = Object.fromEntries(
    Object.entries(document.presentation.node_positions).filter(([n]) => ids.has(n)),
  );
  fragment.presentation.groups = structuredClone(
    document.presentation.groups.filter(
      (g) => g.members.length && g.members.every((n) => ids.has(n)),
    ),
  );
  const boundary = document.authoring.edges.filter(
    (e) => ids.has(e.source_node) !== ids.has(e.target_node),
  );
  return { document: validateDocument(fragment), boundary, removedBindings: bindings };
}

export type Difference = {
  path: string;
  before: unknown;
  after: unknown;
  category: 'semantic' | 'presentation' | 'binding';
};
/** Stable IDs, not labels or array order, identify graph entities. */
export function compareDocuments(before: StudioDocument, after: StudioDocument): Difference[] {
  const out: Difference[] = [];
  function compare(path: string, a: unknown, b: unknown, category: Difference['category']) {
    if (JSON.stringify(a) !== JSON.stringify(b)) out.push({ path, before: a, after: b, category });
  }
  const a = new Map(before.authoring.nodes.map((n) => [n.id, n]));
  const b = new Map(after.authoring.nodes.map((n) => [n.id, n]));
  for (const key of new Set([...a.keys(), ...b.keys()])) {
    const left = a.get(key),
      right = b.get(key);
    if (!left || !right) compare(`nodes.${key}`, left, right, 'semantic');
    else
      for (const field of [
        'label',
        'kind',
        'operation_ref',
        'operation_version',
        'schema_digest',
        'parameter_drafts',
        'input_bindings',
      ] as const)
        compare(
          `nodes.${key}.${field}`,
          left[field],
          right[field],
          field === 'label' ? 'presentation' : field === 'input_bindings' ? 'binding' : 'semantic',
        );
  }
  const leftEdges = new Map(before.authoring.edges.map((e) => [e.id, e]));
  const rightEdges = new Map(after.authoring.edges.map((e) => [e.id, e]));
  for (const key of new Set([...leftEdges.keys(), ...rightEdges.keys()]))
    compare(`edges.${key}`, leftEdges.get(key), rightEdges.get(key), 'semantic');
  const outputs = (d: StudioDocument) =>
    d.authoring.desired_outputs.map((o) => `${o.node_id}:${o.port_id}`).sort();
  compare('desired_outputs', outputs(before), outputs(after), 'semantic');
  compare('host_binding', before.host_binding, after.host_binding, 'binding');
  compare('title', before.identity.title, after.identity.title, 'presentation');
  for (const field of ['node_positions', 'groups', 'viewport'] as const)
    compare(field, before.presentation[field], after.presentation[field], 'presentation');
  return out;
}
