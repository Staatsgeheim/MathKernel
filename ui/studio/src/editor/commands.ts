import {
  type DraftEdge,
  type DraftNode,
  type Position,
  type StudioDocument,
  newId,
  validateDocument,
} from './document';
import type { Operation, Port } from '../host/contracts';

export function descriptor(node: DraftNode, catalog: readonly Operation[]): Operation | undefined {
  return catalog.find(
    (o) =>
      o.operation_ref === node.operation_ref &&
      o.operation_version === node.operation_version &&
      o.schema_digest === node.schema_digest,
  );
}
function port(
  d: StudioDocument,
  catalog: readonly Operation[],
  nodeId: string,
  portId: string,
  direction: 'input' | 'output',
): Port | undefined {
  const n = d.authoring.nodes.find((n) => n.id === nodeId);
  const o = n && descriptor(n, catalog);
  return (
    o &&
    (direction === 'input' ? o.input_ports : o.output_ports).find(
      (p) => p.id === portId && p.direction === direction,
    )
  );
}
export type Compatibility = {
  state: 'compatible' | 'incompatible' | 'requires_check' | 'unavailable';
  reason: string;
};
export function compatibility(
  d: StudioDocument,
  catalog: readonly Operation[],
  e: DraftEdge,
): Compatibility {
  const source = d.authoring.nodes.find((n) => n.id === e.source_node);
  const target = d.authoring.nodes.find((n) => n.id === e.target_node);
  if (!source || !target) return { state: 'incompatible', reason: 'Both endpoints must exist.' };
  if ([source, target].some((n) => descriptor(n, catalog)?.availability === 'unavailable'))
    return { state: 'unavailable', reason: 'An endpoint operation is unavailable.' };
  const a = port(d, catalog, e.source_node, e.source_port, 'output');
  const b = port(d, catalog, e.target_node, e.target_port, 'input');
  if (
    (!a && descriptor(source, catalog)?.composition === 'complete') ||
    (!b && descriptor(target, catalog)?.composition === 'complete')
  )
    return {
      state: 'incompatible',
      reason: 'Port ID or direction is absent from the complete host contract.',
    };
  if (!a || !b)
    return {
      state: 'requires_check',
      reason: 'Port metadata is incomplete or unresolved; host validation required.',
    };
  if (a.type_ref && b.type_ref && a.type_ref !== b.type_ref)
    return {
      state: 'incompatible',
      reason: `Declared types differ: ${a.type_ref} → ${b.type_ref}. No conversion is implied.`,
    };
  if (
    a.shape &&
    b.shape &&
    (a.shape.length !== b.shape.length ||
      a.shape.some(
        (v, i) => typeof v === 'number' && typeof b.shape![i] === 'number' && v !== b.shape![i],
      ))
  )
    return { state: 'incompatible', reason: 'Declared dimensions differ.' };
  if (
    !a.type_ref ||
    !b.type_ref ||
    !a.shape ||
    !b.shape ||
    [...a.shape, ...b.shape].some((v) => typeof v !== 'number')
  )
    return {
      state: 'requires_check',
      reason: 'Shape or type constraints require host validation.',
    };
  return {
    state: 'compatible',
    reason: 'Declared ports match. This is advisory, not mathematical verification.',
  };
}
type EditCommand =
  | { type: 'add'; node: DraftNode; position: Position }
  | { type: 'label'; nodeId: string; label: string }
  | { type: 'title'; title: string }
  | { type: 'parameters'; nodeId: string; drafts: DraftNode['parameter_drafts'] }
  | { type: 'move'; positions: Record<string, Position> }
  | { type: 'viewport'; viewport: StudioDocument['presentation']['viewport'] }
  | { type: 'delete'; nodeIds: string[] }
  | { type: 'duplicate'; nodeIds: string[] }
  | { type: 'connect'; edge: DraftEdge; replace?: boolean }
  | { type: 'reconnect'; edgeId: string; edge: DraftEdge; replace?: boolean }
  | { type: 'disconnect'; edgeId: string }
  | { type: 'output'; nodeId: string; portId: string; selected: boolean };
export type Command = EditCommand & { transaction?: string };
export interface History {
  document: StudioDocument;
  past: StudioDocument[];
  future: StudioDocument[];
  transaction?: string;
}
export function history(document: StudioDocument): History {
  return { document, past: [], future: [] };
}
const MAX_HISTORY_BYTES = 20 * 1024 * 1024;
function bounded(snapshots: StudioDocument[]): StudioDocument[] {
  let size = 0;
  const out: StudioDocument[] = [];
  for (let i = snapshots.length - 1; i >= 0 && out.length < 100; i--) {
    const item = snapshots[i]!;
    size += JSON.stringify(item).length * 2;
    if (size > MAX_HISTORY_BYTES) break;
    out.unshift(item);
  }
  return out;
}
function projection(d: StudioDocument): string {
  return JSON.stringify({
    ...d.authoring,
    revision: 0,
    nodes: d.authoring.nodes.map((n) => ({ ...n, label: '' })),
  });
}
function revise(previous: StudioDocument, next: StudioDocument): StudioDocument {
  next.authoring.revision =
    previous.authoring.revision + (projection(previous) !== projection(next) ? 1 : 0);
  const cosmetic = (d: StudioDocument) =>
    JSON.stringify({
      title: d.identity.title,
      labels: d.authoring.nodes.map((n) => [n.id, n.label]),
      ...d.presentation,
      revision: 0,
    });
  next.presentation.revision =
    previous.presentation.revision + (cosmetic(previous) !== cosmetic(next) ? 1 : 0);
  return validateDocument(next);
}
export function execute(state: History, c: Command, catalog: readonly Operation[] = []): History {
  const d = structuredClone(state.document);
  function node(nodeId: string): DraftNode {
    const n = d.authoring.nodes.find((n) => n.id === nodeId);
    if (!n) throw new Error('Node no longer exists.');
    return n;
  }
  switch (c.type) {
    case 'add':
      d.authoring.nodes.push(structuredClone(c.node));
      d.presentation.node_positions[c.node.id] = c.position;
      break;
    case 'label':
      node(c.nodeId).label = c.label;
      break;
    case 'title':
      d.identity.title = c.title;
      break;
    case 'parameters':
      node(c.nodeId).parameter_drafts = structuredClone(c.drafts);
      break;
    case 'move':
      for (const [key, value] of Object.entries(c.positions)) {
        node(key);
        d.presentation.node_positions[key] = value;
      }
      break;
    case 'viewport':
      d.presentation.viewport = c.viewport;
      break;
    case 'delete': {
      const ids = new Set(c.nodeIds);
      c.nodeIds.forEach(node);
      d.authoring.nodes = d.authoring.nodes.filter((n) => !ids.has(n.id));
      d.authoring.edges = d.authoring.edges.filter(
        (e) => !ids.has(e.source_node) && !ids.has(e.target_node),
      );
      d.authoring.desired_outputs = d.authoring.desired_outputs.filter((o) => !ids.has(o.node_id));
      for (const key of ids) delete d.presentation.node_positions[key];
      d.presentation.groups.forEach((g) => {
        g.members = g.members.filter((n) => !ids.has(n));
      });
      break;
    }
    case 'duplicate': {
      const ids = new Map([...new Set(c.nodeIds)].map((key) => [key, newId()]));
      for (const [old, fresh] of ids) {
        const copy = structuredClone(node(old));
        copy.id = fresh;
        copy.label += ' (copy)';
        // External object bindings are intentionally not retained without rebinding.
        copy.input_bindings = {};
        d.authoring.nodes.push(copy);
        const p = d.presentation.node_positions[old] ?? { x: 0, y: 0 };
        d.presentation.node_positions[fresh] = { x: p.x + 40, y: p.y + 40 };
      }
      const internal = d.authoring.edges.filter(
        (e) => ids.has(e.source_node) && ids.has(e.target_node),
      );
      d.authoring.edges.push(
        ...internal.map((e) => ({
          ...e,
          id: newId(),
          source_node: ids.get(e.source_node)!,
          target_node: ids.get(e.target_node)!,
        })),
      );
      break;
    }
    case 'connect':
    case 'reconnect': {
      if (c.type === 'reconnect') {
        if (c.edge.id !== c.edgeId || !d.authoring.edges.some((e) => e.id === c.edgeId))
          throw new Error('Reconnect requires the original edge ID.');
        d.authoring.edges = d.authoring.edges.filter((e) => e.id !== c.edgeId);
      }
      const check = compatibility(d, catalog, c.edge);
      if (check.state === 'incompatible') throw new Error(check.reason);
      const target = port(d, catalog, c.edge.target_node, c.edge.target_port, 'input');
      const occupied = d.authoring.edges.filter(
        (e) => e.target_node === c.edge.target_node && e.target_port === c.edge.target_port,
      );
      if (occupied.length && target?.cardinality !== 'many') {
        if (!c.replace) throw new Error('Input already connected. Explicit Replace is required.');
        d.authoring.edges = d.authoring.edges.filter((e) => !occupied.includes(e));
      }
      if (
        d.authoring.edges.some(
          (e) =>
            e.source_node === c.edge.source_node &&
            e.source_port === c.edge.source_port &&
            e.target_node === c.edge.target_node &&
            e.target_port === c.edge.target_port,
        )
      )
        throw new Error('Connection already exists.');
      d.authoring.edges.push(structuredClone(c.edge));
      break;
    }
    case 'disconnect':
      d.authoring.edges = d.authoring.edges.filter((e) => e.id !== c.edgeId);
      break;
    case 'output': {
      node(c.nodeId);
      d.authoring.desired_outputs = d.authoring.desired_outputs.filter(
        (o) => o.node_id !== c.nodeId || o.port_id !== c.portId,
      );
      if (c.selected) d.authoring.desired_outputs.push({ node_id: c.nodeId, port_id: c.portId });
      break;
    }
  }
  const next = revise(state.document, d);
  if (JSON.stringify(next) === JSON.stringify(state.document)) return state;
  return {
    document: next,
    past:
      c.transaction && c.transaction === state.transaction
        ? state.past
        : bounded([...state.past, state.document]),
    future: [],
    transaction: c.transaction,
  };
}
export function undo(state: History): History {
  const previous = state.past.at(-1);
  if (!previous) return state;
  return {
    document: revise(state.document, structuredClone(previous)),
    past: state.past.slice(0, -1),
    future: bounded([...state.future, state.document]),
  };
}
export function redo(state: History): History {
  const next = state.future.at(-1);
  if (!next) return state;
  return {
    document: revise(state.document, structuredClone(next)),
    past: bounded([...state.past, state.document]),
    future: state.future.slice(0, -1),
  };
}
export interface Diagnostic {
  code: string;
  message: string;
  nodeId?: string;
  edgeId?: string;
  revision: number;
  layer: 'editor' | 'advisory';
}
export function diagnostics(d: StudioDocument, catalog: readonly Operation[]): Diagnostic[] {
  const out: Diagnostic[] = [];
  for (const n of d.authoring.nodes) {
    const op = descriptor(n, catalog);
    if (n.operation_ref && !op)
      out.push({
        code: 'UNRESOLVED_OPERATION',
        message: `${n.label}: operation or schema version unresolved.`,
        nodeId: n.id,
        revision: d.authoring.revision,
        layer: 'advisory',
      });
    else if (op && op.composition !== 'complete')
      out.push({
        code: 'PARTIAL_METADATA',
        message: `${n.label}: ${op.coverage}`,
        nodeId: n.id,
        revision: d.authoring.revision,
        layer: 'advisory',
      });
  }
  for (const e of d.authoring.edges) {
    const c = compatibility(d, catalog, e);
    if (c.state !== 'compatible')
      out.push({
        code: c.state.toUpperCase(),
        message: c.reason,
        edgeId: e.id,
        nodeId: e.target_node,
        revision: d.authoring.revision,
        layer: 'advisory',
      });
  }
  // Kahn's algorithm is a structural cycle check, never a launch order.
  const incoming = new Map(d.authoring.nodes.map((n) => [n.id, 0]));
  const adjacency = new Map<string, string[]>();
  for (const e of d.authoring.edges) {
    incoming.set(e.target_node, incoming.get(e.target_node)! + 1);
    const list = adjacency.get(e.source_node) ?? [];
    list.push(e.target_node);
    adjacency.set(e.source_node, list);
  }
  const ready = [...incoming].filter(([, count]) => count === 0).map(([key]) => key);
  for (let i = 0; i < ready.length; i++)
    for (const to of adjacency.get(ready[i]!) ?? []) {
      incoming.set(to, incoming.get(to)! - 1);
      if (!incoming.get(to)) ready.push(to);
    }
  if (ready.length !== d.authoring.nodes.length)
    out.push({
      code: 'CYCLE',
      message: 'Cycle retained in the draft. No host control semantics have been established.',
      revision: d.authoring.revision,
      layer: 'editor',
    });
  return out;
}
