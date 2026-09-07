import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import {
  emptyDocument,
  importDocument,
  makeInput,
  serializeDocument,
  type DraftEdge,
  type DraftNode,
} from '../src/editor/document';
import { compatibility, diagnostics, execute, history, redo, undo } from '../src/editor/commands';
import { parseJson } from '../src/security/json';
import { integerError, rationalError, realError } from '../src/inspectors/exact';
import type { Operation } from '../src/host/contracts';
const fixture = readFileSync(
  new URL('../design/examples/exact_input_fixture.mkstudio.json', import.meta.url),
  'utf8',
);
function graph() {
  return history(importDocument(fixture));
}
function operation(type = 'Integer', shape: (number | string | null)[] = []): Operation {
  return {
    descriptor_id: 'test',
    operation_ref: 'fixture.integer_input',
    operation_version: 'fixture-1',
    schema_digest: 'a'.repeat(64),
    title: 'Fixture',
    domain: 'test',
    description: 'Fixture',
    availability: 'available',
    composition: 'complete',
    input_ports: [
      {
        id: 'value',
        direction: 'input',
        label: 'Value',
        type_ref: type,
        cardinality: 'one',
        required: true,
        shape,
      },
    ],
    output_ports: [
      {
        id: 'value',
        direction: 'output',
        label: 'Value',
        type_ref: type,
        cardinality: 'one',
        required: true,
        shape,
      },
    ],
    input_types: [],
    output_types: [],
    engines: [],
    trust_levels: [],
    verification_methods: [],
    parameter_schema: {},
    coverage: 'test',
  };
}
describe('UI exactness and inert document acceptance', () => {
  it('roundtrips the supplied exact fixture without 2^53 coercion', () => {
    const d = importDocument(fixture);
    expect(d.authoring.nodes[0]!.parameter_drafts.value!.text).toBe('9007199254740993');
    expect(importDocument(serializeDocument(d))).toEqual(d);
  });
  it.each(['integer', 'rational', 'real'] as const)(
    'serializes %s MathIR-shaped input as strings',
    (kind) => {
      const n = makeInput(kind);
      const d = emptyDocument();
      d.authoring.nodes.push(n);
      expect(importDocument(serializeDocument(d)).authoring.nodes[0]).toEqual(n);
    },
  );
  it.each([
    '{"x":1,"x":2}',
    '{"x":1,"\\u0078":2}',
    '{"__proto__":{}}',
    '{"constructor":1}',
    '{"prototype":1}',
    '{"v":9007199254740993}',
    '[1,]',
    '{"x":1,}',
    'NaN',
    '1e999',
    '{} trailing',
  ])('rejects hostile/invalid JSON: %s', (input) => expect(() => parseJson(input)).toThrow());
  it('rejects excessive nesting before parsing the full object', () =>
    expect(() => parseJson('['.repeat(34) + '0' + ']'.repeat(34))).toThrow('complexity'));
  it('checks UTF-8 bytes, not only character count', () =>
    expect(() => parseJson('"éééé"', 8)).toThrow('byte limit'));
  it('rejects unsupported critical schema versions', () =>
    expect(() => importDocument(fixture.replace('mk.studio/1', 'mk.studio/2'))).toThrow());
  it('rejects imported trust and authorization fields', () => {
    const d = JSON.parse(fixture);
    d.approved = true;
    d.trust = 'formal';
    expect(() => importDocument(JSON.stringify(d))).toThrow();
  });
  it('preserves hostile labels as inert text', () => {
    const d = importDocument(fixture);
    d.authoring.nodes[0]!.label = '<script>FORMAL approved</script>';
    expect(importDocument(serializeDocument(d)).authoring.nodes[0]!.label).toContain('<script>');
  });
  it('rejects duplicate node IDs and dangling edges', () => {
    const d = importDocument(fixture);
    d.authoring.nodes[1]!.id = d.authoring.nodes[0]!.id;
    expect(() => serializeDocument(d)).toThrow('Duplicate');
  });
  it('rejects dangling desired outputs', () => {
    const d = importDocument(fixture);
    d.authoring.desired_outputs[0]!.node_id = 'absent';
    expect(() => serializeDocument(d)).toThrow('Dangling');
  });
  it('rejects total inline text over one MiB per node', () => {
    const d = importDocument(fixture);
    d.authoring.nodes[0]!.parameter_drafts = {
      a: { encoding: 'text', text: 'a'.repeat(600000) },
      b: { encoding: 'text', text: 'a'.repeat(600000) },
    };
    expect(() => serializeDocument(d)).toThrow('1 MiB');
  });
  it('accepts a 50,000-digit integer and rational without conversion', () => {
    const huge = '9'.repeat(50000);
    expect(integerError(huge)).toBeNull();
    expect(rationalError(huge, '7')).toBeNull();
  });
  it.each(['0', '-0', '+00'])('rejects zero denominator %s', (value) =>
    expect(rationalError('1', value)).toMatch(/zero/),
  );
  it('distinguishes blank from zero and locale comma from decimal point', () => {
    expect(integerError('')).not.toBeNull();
    expect(integerError('0')).toBeNull();
    expect(realError('0,1')).toMatch(/comma/);
    expect(realError('NaN')).not.toBeNull();
    expect(realError('1e-16')).toBeNull();
  });
});
describe('shared command semantics', () => {
  it('renaming and moving change presentation only', () => {
    let s = graph();
    s = execute(s, { type: 'label', nodeId: 'n-input', label: 'Renamed' });
    s = execute(s, { type: 'move', positions: { 'n-input': { x: 10, y: 20 } } });
    expect(s.document.authoring.revision).toBe(1);
    expect(s.document.presentation.revision).toBe(3);
    expect(s.document.authoring.edges[0]!.source_node).toBe('n-input');
  });
  it('invalid parameter drafts remain recoverable and undoable', () => {
    const s = execute(graph(), {
      type: 'parameters',
      nodeId: 'n-input',
      drafts: { value: { encoding: 'json_text', text: '{"kind":' } },
    });
    expect(
      importDocument(serializeDocument(s.document)).authoring.nodes[0]!.parameter_drafts.value!
        .text,
    ).toBe('{"kind":');
    const u = undo(s);
    expect(u.document.authoring.nodes[0]!.parameter_drafts.value!.text).toBe('9007199254740993');
    expect(u.document.authoring.revision).toBe(3);
    expect(redo(u).document.authoring.revision).toBe(4);
  });
  it('deletes incident wires and outputs in one transaction and restores IDs on undo', () => {
    const s = graph(),
      deleted = execute(s, { type: 'delete', nodeIds: ['n-inspect'] });
    expect(deleted.document.authoring.edges).toEqual([]);
    expect(deleted.document.authoring.desired_outputs).toEqual([]);
    expect(undo(deleted).document.authoring.nodes).toEqual(s.document.authoring.nodes);
    expect(undo(deleted).document.authoring.edges).toEqual(s.document.authoring.edges);
  });
  it('duplicates with fresh IDs and remaps only internal edges', () => {
    const s = execute(graph(), { type: 'duplicate', nodeIds: ['n-input', 'n-inspect'] });
    const n = s.document.authoring.nodes;
    expect(new Set(n.map((n) => n.id)).size).toBe(4);
    expect(s.document.authoring.edges[1]!.source_node).toBe(n[2]!.id);
    expect(s.document.authoring.edges[1]!.target_node).toBe(n[3]!.id);
    expect(
      execute(graph(), { type: 'duplicate', nodeIds: ['n-input'] }).document.authoring.edges,
    ).toHaveLength(1);
  });
  it('does not retain external object authority when duplicating', () => {
    const s = graph();
    s.document.authoring.nodes[0]!.input_bindings.x = {
      host_instance_id: 'a',
      workspace_id: 'w',
      object_id: 'o',
      resolution: 'unresolved',
    };
    const d = execute(s, { type: 'duplicate', nodeIds: ['n-input'] });
    expect(d.document.authoring.nodes[2]!.input_bindings).toEqual({});
  });
  it('leaves the original edge intact on failed reconnect', () => {
    const s = graph();
    const e = { ...s.document.authoring.edges[0]!, target_node: 'missing' };
    expect(() => execute(s, { type: 'reconnect', edgeId: e.id, edge: e })).toThrow();
    expect(s.document.authoring.edges[0]!.target_node).toBe('n-inspect');
  });
  it('requires explicit Replace for occupied input with unknown cardinality', () => {
    const s = graph();
    const n = makeInput('integer');
    const a = execute(s, { type: 'add', node: n, position: { x: 0, y: 0 } });
    const e = { ...s.document.authoring.edges[0]!, id: 'replacement', source_node: n.id };
    expect(() => execute(a, { type: 'connect', edge: e })).toThrow('Replace');
    const b = execute(a, { type: 'connect', edge: e, replace: true });
    expect(b.document.authoring.edges).toEqual([e]);
  });
  it('does not silently migrate a changed catalog schema', () => {
    const s = graph();
    expect(
      diagnostics(s.document, [operation()]).some((p) => p.code === 'UNRESOLVED_OPERATION'),
    ).toBe(true);
  });
  it('reports unsupported cycles without removing them', () => {
    const s = graph();
    const e = {
      id: 'back',
      source_node: 'n-inspect',
      source_port: 'result',
      target_node: 'n-input',
      target_port: 'value',
    };
    const d = execute(s, { type: 'connect', edge: e });
    expect(diagnostics(d.document, []).some((p) => p.code === 'CYCLE')).toBe(true);
    expect(d.document.authoring.edges).toHaveLength(2);
  });
  it('bounds undo history', () => {
    let s = graph();
    for (let i = 0; i < 120; i++)
      s = execute(s, { type: 'label', nodeId: 'n-input', label: String(i) });
    expect(s.past).toHaveLength(100);
  });
  it('undo/redo roundtrips a varied command sequence without reusing semantic revisions', () => {
    let s = graph();
    const original = serializeDocument(s.document);
    for (let i = 0; i < 25; i++) {
      s = execute(s, {
        type: 'parameters',
        nodeId: 'n-input',
        drafts: { value: { encoding: 'text', text: `${i}9007199254740993` } },
      });
      s = execute(s, { type: 'move', positions: { 'n-input': { x: i * 3, y: i * 5 } } });
    }
    const last = s.document;
    for (let i = 0; i < 50; i++) s = undo(s);
    const restored = {
      ...s.document,
      authoring: { ...s.document.authoring, revision: 1 },
      presentation: { ...s.document.presentation, revision: 1 },
    };
    expect(serializeDocument(restored)).toBe(original);
    for (let i = 0; i < 50; i++) s = redo(s);
    expect(s.document.authoring.nodes).toEqual(last.authoring.nodes);
    expect(s.document.authoring.revision).toBeGreaterThan(last.authoring.revision);
  });
  it.each([
    ['Integer', [], 'compatible'],
    ['Real64', [], 'incompatible'],
    ['Integer', [3, 4], 'incompatible'],
    ['Integer', ['n'], 'incompatible'],
  ] as const)('uses declared metadata only: %s %j', (type, shape, expected) => {
    const s = graph();
    const a = operation();
    const b = {
      ...operation(type, [...shape]),
      operation_ref: 'fixture.inspect',
      descriptor_id: 'second',
    };
    s.document.authoring.nodes.forEach((n) => (n.schema_digest = 'a'.repeat(64)));
    expect(compatibility(s.document, [a, b], s.document.authoring.edges[0]!).state).toBe(expected);
  });
  it('symbolic dimensions of equal rank require a host check', () => {
    const s = graph();
    const a = operation('Integer', [3]);
    const b = { ...operation('Integer', ['n']), operation_ref: 'fixture.inspect' };
    s.document.authoring.nodes.forEach((n) => (n.schema_digest = 'a'.repeat(64)));
    expect(compatibility(s.document, [a, b], s.document.authoring.edges[0]!).state).toBe(
      'requires_check',
    );
  });
});
it('coalesces live field edits into one undo step while retaining intermediate draft text', () => {
  const start = graph();
  let s = start;
  for (const text of ['9', '90', '9007199254740993', '9007199254740993x'])
    s = execute(s, {
      type: 'parameters',
      nodeId: 'n-input',
      drafts: { value: { encoding: 'text', text } },
      transaction: 'field-session-1',
    });
  expect(s.past).toHaveLength(1);
  expect(s.document.authoring.nodes[0]!.parameter_drafts.value!.text).toBe('9007199254740993x');
  expect(undo(s).document.authoring.nodes).toEqual(start.document.authoring.nodes);
  s = execute(s, {
    type: 'parameters',
    nodeId: 'n-input',
    drafts: { value: { encoding: 'text', text: 'new edit' } },
    transaction: 'field-session-2',
  });
  expect(s.past).toHaveLength(2);
});
