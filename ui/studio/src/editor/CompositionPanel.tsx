import { useMemo, useState } from 'react';
import { Dialog, Field, JsonView, Text } from '../app/components';
import { newId, serializeDocument, type StudioDocument } from './document';
import { compareDocuments, selectionFragment } from './composition';
import type { Command, History } from './commands';
import type { Operation } from '../host/contracts';
import { descriptor } from './commands';

export function downloadJson(value: unknown, name: string) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob),
    link = document.createElement('a');
  link.href = url;
  link.download = name.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 100);
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function CompositionPanel({
  state,
  selection,
  catalog,
  command,
  onClose,
}: {
  state: History;
  selection: string[];
  catalog: Operation[];
  command: (c: Command) => boolean;
  onClose: () => void;
}) {
  const [name, setName] = useState('Reusable selection');
  const [fragment, setFragment] = useState<ReturnType<typeof selectionFragment> | null>(null);
  const [compareIndex, setCompareIndex] = useState(Math.max(0, state.past.length - 1));
  const [collapsed, setCollapsed] = useState<string[]>([]);
  const differences = useMemo(
    () =>
      state.past[compareIndex] ? compareDocuments(state.past[compareIndex]!, state.document) : [],
    [state, compareIndex],
  );
  const doc = state.document;
  return (
    <Dialog title="Groups, fragments and revision comparison" onClose={onClose}>
      <section>
        <h3>Visual groups</h3>
        <p>Frames organize nodes. Their names never select compute targets or change execution.</p>
        <Field label="Group / fragment title" value={name} onCommit={setName} />
        <button
          disabled={!selection.length}
          onClick={() =>
            command({ type: 'group', group: { id: newId(), title: name, members: selection } })
          }
        >
          Group selected nodes
        </button>
        {doc.presentation.groups.map((g) => {
          const boundary = doc.authoring.edges.filter(
            (e) => g.members.includes(e.source_node) !== g.members.includes(e.target_node),
          );
          const unresolved = g.members.filter((id) => {
            const n = doc.authoring.nodes.find((n) => n.id === id)!;
            return n.operation_ref && !descriptor(n, catalog);
          }).length;
          return (
            <article key={g.id} className="group-card">
              <Field
                label="Frame title"
                value={g.title}
                onCommit={(title) => command({ type: 'group', group: { ...g, title } })}
              />
              <p>
                {g.members.length} members · {boundary.length} crossing connections · {unresolved}{' '}
                unresolved operations · no run status
              </p>
              <div className="toolbar">
                <button
                  onClick={() =>
                    setCollapsed((old) =>
                      old.includes(g.id) ? old.filter((id) => id !== g.id) : [...old, g.id],
                    )
                  }
                >
                  {collapsed.includes(g.id) ? 'Expand member list' : 'Collapse member list'}
                </button>
                <button
                  onClick={() =>
                    command({
                      type: 'move',
                      positions: Object.fromEntries(
                        g.members.map((id) => {
                          const p = doc.presentation.node_positions[id] ?? { x: 0, y: 0 };
                          return [id, { x: p.x + 40, y: p.y + 40 }];
                        }),
                      ),
                    })
                  }
                >
                  Move frame and members +40
                </button>
                <button onClick={() => command({ type: 'ungroup', groupId: g.id })}>
                  Remove frame; keep nodes
                </button>
              </div>
              {!collapsed.includes(g.id) && (
                <ul>
                  {g.members.map((id) => (
                    <li key={id}>
                      <Text>{doc.authoring.nodes.find((n) => n.id === id)!.label}</Text>
                    </li>
                  ))}
                </ul>
              )}
            </article>
          );
        })}
      </section>
      <section>
        <h3>Reusable authoring fragment</h3>
        <p>
          Exports only the selected nodes and internal wiring. Host bindings and results are
          excluded. Import the file and choose Insert as fragment to make a fresh copy.
        </p>
        <button
          disabled={!selection.length}
          onClick={() => setFragment(selectionFragment(doc, selection, name))}
        >
          Review selected fragment
        </button>
        {fragment && (
          <div className="notice">
            <p>
              {fragment.document.authoring.nodes.length} nodes · {fragment.boundary.length} external
              connections omitted · {fragment.removedBindings.length} object bindings removed
            </p>
            <JsonView
              value={{
                external_connections: fragment.boundary,
                removed_bindings: fragment.removedBindings,
              }}
            />
            <button
              onClick={() => {
                serializeDocument(fragment.document);
                downloadJson(fragment.document, `${name}.mkstudio.json`);
              }}
            >
              Export reviewed fragment
            </button>
          </div>
        )}
      </section>
      <section>
        <h3>Compare retained revision with current draft</h3>
        <p>
          History is bounded to 100 edits and 20 MiB in this tab. Compare shows changes; it never
          reexecutes or restores a run.
        </p>
        <label className="field">
          Earlier snapshot
          <select value={compareIndex} onChange={(e) => setCompareIndex(Number(e.target.value))}>
            {state.past.map((d, i) => (
              <option key={i} value={i}>
                Draft {d.authoring.revision}, layout {d.presentation.revision}
              </option>
            ))}
          </select>
        </label>
        <p>{differences.length} changed fields</p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Category / field</th>
                <th>Earlier</th>
                <th>Current</th>
              </tr>
            </thead>
            <tbody>
              {differences.slice(0, 200).map((d) => (
                <tr key={d.path}>
                  <td>
                    {d.category}
                    <br />
                    <Text>{d.path}</Text>
                  </td>
                  <td>
                    <JsonView value={d.before} />
                  </td>
                  <td>
                    <JsonView value={d.after} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {differences.length > 200 && (
          <p>Display limited to 200 changes. Export includes all changes.</p>
        )}
        <button
          disabled={!state.past.length}
          onClick={() =>
            downloadJson(
              {
                schema: 'mk.studio.diff/1',
                before: state.past[compareIndex]?.identity,
                after: doc.identity,
                changes: differences,
              },
              'studio-revision-diff.json',
            )
          }
        >
          Export comparison
        </button>
      </section>
    </Dialog>
  );
}

export function insertionNotices(fragment: StudioDocument, catalog: Operation[]) {
  return {
    unresolved_operations: fragment.authoring.nodes
      .filter((n) => n.operation_ref && !descriptor(n, catalog))
      .map((n) => n.operation_ref),
    removed_object_bindings: fragment.authoring.nodes.reduce(
      (s, n) => s + Object.keys(n.input_bindings).length,
      0,
    ),
  };
}
