import { describe, expect, it } from 'vitest';
import { checkReply, readPending, retainPending } from '../src/host/pending';
import { acceptSnapshot, runSchema } from '../src/host/workflow';
import { projectAudio } from '../src/viewers/audio';
import { compareDocuments } from '../src/editor/composition';
import { emptyDocument } from '../src/editor/document';

const run = () =>
  runSchema.parse({
    run_ref: 'r',
    binding: { document_id: 'd', draft_revision: 1, document_digest: 'a'.repeat(64) },
    plan_ref: 'p',
    revision: 1,
    cursor: 'c',
    observed_at: '2026-09-07T00:00:00Z',
    execution: 'running',
    verification: 'unknown',
    artifacts: 'unknown',
    resources: 'unknown',
    cost: 'unknown',
    warnings: [],
    attempts: [],
    events: [],
    actions: [],
  });
const event = {
  event_id: 'event',
  revision: 1,
  kind: 'progress',
  message: 'one',
  observed_at: '2026-09-07T00:00:00Z',
};
const attempt = {
  node_id: 'node',
  attempt_id: 'attempt',
  revision: 1,
  state: 'running',
  result_ref: null,
  progress: null,
  progress_kind: 'indeterminate' as const,
  observed_at: event.observed_at,
};
describe('U4 observation and recovery contract', () => {
  it('rejects conflicting event IDs even in the first observation', () => {
    expect(() =>
      acceptSnapshot(null, { ...run(), events: [event, { ...event, message: 'changed' }] }),
    ).toThrow('Conflicting');
    expect(acceptSnapshot(null, { ...run(), events: [event, event] }).events).toHaveLength(1);
  });
  it('rejects duplicate attempts and changed facts at a fixed attempt revision', () => {
    expect(() => acceptSnapshot(null, { ...run(), attempts: [attempt, attempt] })).toThrow(
      'Duplicate',
    );
    expect(() =>
      acceptSnapshot(
        { ...run(), attempts: [attempt] },
        { ...run(), revision: 2, attempts: [{ ...attempt, state: 'done' }] },
      ),
    ).toThrow('Conflicting attempt');
  });
  it('rejects changed event facts across snapshots', () => {
    expect(() =>
      acceptSnapshot(
        { ...run(), events: [event] },
        { ...run(), revision: 2, events: [{ ...event, message: 'changed' }] },
      ),
    ).toThrow('Conflicting event');
  });
  it('retains only command identity, refuses overwrites and validates plan during reconciliation', () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (k: string) => values.get(k) ?? null,
      setItem: (k: string, v: string) => {
        values.set(k, v);
      },
    };
    retainPending(storage, 'key', { request: 'q', plan: 'p' });
    expect(readPending(storage, 'key')).toEqual({ request: 'q', plan: 'p' });
    expect(() => retainPending(storage, 'key', { request: 'new', plan: 'p' })).toThrow(
      'unresolved',
    );
    expect(() =>
      checkReply(
        {
          client_request_id: 'q',
          plan_ref: 'other',
          outcome: 'accepted',
          run_ref: 'r',
          message: '',
        },
        { request: 'q', plan: 'p' },
      ),
    ).toThrow('mismatch');
    expect(() =>
      checkReply(
        { client_request_id: 'q', plan_ref: 'p', outcome: 'accepted', run_ref: null, message: '' },
        { request: 'q', plan: 'p' },
      ),
    ).toThrow('mismatch');
    expect(values.get('key')).not.toContain('authority');
    values.set('key', 'old-request');
    expect(readPending(storage, 'key')?.request).toBe('old-request');
    values.set('key', '{broken');
    expect(() => readPending(storage, 'key')).toThrow();
  });
  it('storage write failures refuse submission instead of forgetting pending work', () => {
    expect(() =>
      retainPending({ getItem: () => null, setItem: () => {} }, 'k', { request: 'q', plan: 'p' }),
    ).toThrow('Nothing was submitted');
  });
});
describe('U5 semantic comparison', () => {
  it('does not report reordered edges or output sets as mathematical changes', () => {
    const a = emptyDocument();
    a.authoring.edges = [
      { id: 'a', source_node: 'n', source_port: 'out', target_node: 'm', target_port: 'in' },
      { id: 'b', source_node: 'm', source_port: 'out', target_node: 'o', target_port: 'in' },
    ];
    a.authoring.desired_outputs = [
      { node_id: 'm', port_id: 'out' },
      { node_id: 'o', port_id: 'out' },
    ];
    const b = structuredClone(a);
    b.authoring.edges.reverse();
    b.authoring.desired_outputs.reverse();
    expect(compareDocuments(a, b)).toEqual([]);
    b.authoring.edges[0]!.source_port = 'other';
    expect(compareDocuments(a, b)[0]!.path).toBe('edges.b');
  });
});
describe('U6 resolved audio audition', () => {
  const audio = () => ({
    artifact_schema: 'mathkernel-sonify/1.0',
    schema_version: '1.0',
    title: 'Source',
    tracks: [
      {
        label: 'Track',
        gain: 1,
        pan: 0,
        mapping_refs: [],
        events: [{ label: 'Event', time: 0, duration: 1, values: { frequency: 440, gain: 0.5 } }],
      },
    ],
  });
  it('projects existing events and strips evidence labels', () => {
    const p = projectAudio({ ...audio(), trust: 'formal', approved: true })!;
    expect(p.view?.events[0]?.frequency).toBe(440);
    expect(p.view).not.toHaveProperty('trust');
    expect(p.view).not.toHaveProperty('approved');
  });
  it('refuses unresolved mappings, phase, frequencies and durations instead of silently altering them', () => {
    const mapped = audio();
    mapped.tracks[0]!.mapping_refs.push('map' as never);
    expect(projectAudio(mapped)?.view).toBeNull();
    const long = audio();
    long.tracks[0]!.events[0]!.duration = 61;
    expect(projectAudio(long)?.view).toBeNull();
    const low = audio();
    low.tracks[0]!.events[0]!.values.frequency = 0;
    expect(projectAudio(low)?.view).toBeNull();
    const phase = audio();
    Object.assign(phase.tracks[0]!.events[0]!.values, { phase: 1 });
    expect(projectAudio(phase)?.view).toBeNull();
  });
});
