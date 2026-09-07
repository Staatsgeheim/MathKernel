import { describe, expect, it } from 'vitest';
import {
  emptyDocument,
  makeInput,
  newId,
  serializeDocument,
  importDocument,
} from '../src/editor/document';
import { execute, history, undo } from '../src/editor/commands';
import { selectionFragment, compareDocuments } from '../src/editor/composition';
import { matrixCellError } from '../src/inspectors/StructuredInput';
import { freezeDocument } from '../src/host/workflow';

describe('composition and precision boundaries', () => {
  function graph() {
    let state = history(emptyDocument());
    for (let i = 0; i < 3; i++)
      state = execute(state, {
        type: 'add',
        node: makeInput('integer'),
        position: { x: i * 300, y: 0 },
      });
    const [a, b, c] = state.document.authoring.nodes;
    state = execute(state, {
      type: 'connect',
      edge: {
        id: newId(),
        source_node: a!.id,
        source_port: 'out',
        target_node: b!.id,
        target_port: 'in',
      },
    });
    state = execute(state, {
      type: 'connect',
      edge: {
        id: newId(),
        source_node: b!.id,
        source_port: 'out',
        target_node: c!.id,
        target_port: 'in',
      },
    });
    return state;
  }
  it('groups and member movement change presentation only; frame deletion retains nodes', () => {
    const start = graph(),
      group = {
        id: newId(),
        title: 'GPU is just a label',
        members: start.document.authoring.nodes.map((n) => n.id),
      };
    const grouped = execute(start, { type: 'group', group });
    expect(grouped.document.authoring.revision).toBe(start.document.authoring.revision);
    const removed = execute(grouped, { type: 'ungroup', groupId: group.id });
    expect(removed.document.authoring).toEqual(start.document.authoring);
    expect(undo(removed).document.presentation.groups).toEqual([group]);
  });
  it('fragment export discloses external edges and strips object authority', () => {
    let state = graph();
    const a = state.document.authoring.nodes[0]!;
    state = execute(state, {
      type: 'bindings',
      nodeId: a.id,
      bindings: {
        in: {
          host_instance_id: 'host',
          workspace_id: 'workspace',
          object_id: 'private',
          resolution: 'unresolved',
        },
      },
    });
    const f = selectionFragment(
      state.document,
      state.document.authoring.nodes.slice(0, 2).map((n) => n.id),
      'Fragment',
    );
    expect(f.boundary).toHaveLength(1);
    expect(f.removedBindings).toHaveLength(1);
    expect(f.document.host_binding).toBeNull();
    expect(f.document.authoring.nodes[0]!.input_bindings).toEqual({});
    expect(f.document.authoring.edges).toHaveLength(1);
  });
  it('insertion remaps every internal ID, keeps exact values and is one undo', () => {
    const original = graph(),
      f = selectionFragment(
        original.document,
        original.document.authoring.nodes.map((n) => n.id),
        'Fragment',
      ).document;
    const next = execute(original, { type: 'insert', document: f });
    const inserted = next.document.authoring.nodes.slice(3);
    expect(
      inserted.every((n) => !original.document.authoring.nodes.some((old) => old.id === n.id)),
    ).toBe(true);
    expect(new Set(next.document.authoring.nodes.map((n) => n.id)).size).toBe(6);
    expect(inserted[0]!.parameter_drafts).toEqual(
      original.document.authoring.nodes[0]!.parameter_drafts,
    );
    expect(undo(next).document.authoring.nodes).toEqual(original.document.authoring.nodes);
  });
  it('matrix cells preserve invalid drafts and values beyond Number precision through file roundtrip', () => {
    const d = emptyDocument(),
      n = makeInput('matrix');
    n.parameter_drafts.value!.text = JSON.stringify({
      kind: 'matrix',
      arithmetic: 'exact',
      cells: [
        ['9007199254740993123456789', '1/3'],
        ['', '1/0'],
      ],
    });
    d.authoring.nodes.push(n);
    expect(importDocument(serializeDocument(d)).authoring.nodes[0]!.parameter_drafts).toEqual(
      n.parameter_drafts,
    );
    expect(matrixCellError('9007199254740993123456789', 'exact')).toBeNull();
    expect(matrixCellError('1/0', 'exact')).toContain('zero');
    expect(matrixCellError('', 'exact')).toContain('Blank');
  });
  it('diff shows semantic coefficient changes separately from labels and positions', () => {
    const start = graph(),
      node = start.document.authoring.nodes[0]!;
    let changed = execute(start, { type: 'label', nodeId: node.id, label: 'Renamed' });
    changed = execute(changed, {
      type: 'parameters',
      nodeId: node.id,
      drafts: { x: { encoding: 'text', text: 'x^2+2' } },
    });
    expect(compareDocuments(start.document, changed.document).map((d) => d.category)).toContain(
      'semantic',
    );
    expect(compareDocuments(start.document, changed.document).map((d) => d.category)).toContain(
      'presentation',
    );
  });
  it('frozen digest changes with semantics but not a frame label, position or document title', async () => {
    const start = graph(),
      original = await freezeDocument(start.document);
    const moved = execute(start, {
      type: 'move',
      positions: { [start.document.authoring.nodes[0]!.id]: { x: 123, y: 456 } },
    });
    expect(await freezeDocument(moved.document)).toEqual(original);
    const semantic = execute(start, {
      type: 'parameters',
      nodeId: start.document.authoring.nodes[0]!.id,
      drafts: { x: { encoding: 'text', text: '2' } },
    });
    expect((await freezeDocument(semantic.document)).document_digest).not.toBe(
      original.document_digest,
    );
  });
});
