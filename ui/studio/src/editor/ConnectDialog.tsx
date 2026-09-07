import { useState } from 'react';
import { Dialog, Text } from '../app/components';
import { type DraftEdge, type StudioDocument, newId } from './document';
import { compatibility, descriptor } from './commands';
import type { Operation } from '../host/contracts';
export function ConnectDialog({
  document,
  catalog,
  initial,
  oldId,
  onClose,
  onApply,
}: {
  document: StudioDocument;
  catalog: readonly Operation[];
  initial?: DraftEdge;
  oldId?: string;
  onClose: () => void;
  onApply: (edge: DraftEdge, replace: boolean, oldId?: string) => boolean;
}) {
  const [edge, setEdge] = useState<DraftEdge>(
    initial ?? {
      id: newId(),
      source_node: document.authoring.nodes[0]?.id ?? '',
      source_port: '',
      target_node: document.authoring.nodes[1]?.id ?? '',
      target_port: '',
    },
  );
  const [replace, setReplace] = useState(false);
  const state = compatibility(document, catalog, edge);
  return (
    <Dialog
      title={oldId ? 'Reconnect — original wire stays until confirmed' : 'Connect nodes'}
      onClose={onClose}
    >
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (onApply(edge, replace, oldId)) onClose();
        }}
      >
        <div className="two-columns">
          {(['source', 'target'] as const).map((side) => {
            const key = `${side}_node` as const;
            const pkey = `${side}_port` as const;
            const n = document.authoring.nodes.find((n) => n.id === edge[key]);
            const op = n && descriptor(n, catalog);
            const ports = side === 'source' ? op?.output_ports : op?.input_ports;
            return (
              <section key={side}>
                <label className="field">
                  {side === 'source' ? 'Source node' : 'Destination node'}
                  <select
                    value={edge[key]}
                    onChange={(e) => setEdge({ ...edge, [key]: e.target.value, [pkey]: '' })}
                    required
                  >
                    <option value="">Choose node</option>
                    {document.authoring.nodes.map((n) => (
                      <option value={n.id} key={n.id}>
                        <Text>{n.label}</Text>
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  {side === 'source' ? 'Output port ID' : 'Input port ID'}
                  {ports?.length ? (
                    <select
                      required
                      value={edge[pkey]}
                      onChange={(e) => setEdge({ ...edge, [pkey]: e.target.value })}
                    >
                      <option value="">Choose port</option>
                      {ports.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.label} · {p.type_ref ?? 'unknown'}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      required
                      value={edge[pkey]}
                      onChange={(e) => setEdge({ ...edge, [pkey]: e.target.value })}
                      maxLength={128}
                      placeholder="Explicit unresolved port ID"
                    />
                  )}
                </label>
                {!ports?.length && (
                  <p className="muted">
                    No port contract is available. A manually entered reference stays unresolved.
                  </p>
                )}
              </section>
            );
          })}
        </div>
        <p role="status" className={state.state === 'incompatible' ? 'warning' : 'notice'}>
          {state.state.replace('_', ' ')}: {state.reason}
        </p>
        <label className="check">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          Replace any existing connection on this input
        </label>
        <footer className="dialog-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button className="primary" disabled={state.state === 'incompatible'}>
            Confirm connection
          </button>
        </footer>
      </form>
    </Dialog>
  );
}
