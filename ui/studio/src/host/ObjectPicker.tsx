import { useState } from 'react';
import { Dialog, Field, JsonView, Text } from '../app/components';
import type { HostCommandService } from './client';
import type { Handshake } from './contracts';
import type { DraftNode } from '../editor/document';
import { id } from '../editor/document';
import type { Command } from '../editor/commands';

export function ObjectPicker({
  client,
  host,
  node,
  command,
  onClose,
  onHostError,
}: {
  client: HostCommandService;
  host: Handshake;
  node?: DraftNode;
  command: (c: Command) => boolean;
  onClose: () => void;
  onHostError: (e: unknown) => void;
}) {
  const [objects, setObjects] = useState<
      { object_id: string; revision: string; type_ref: string; summary: string; shape: unknown }[]
    >([]),
    [offset, setOffset] = useState<number | null>(0);
  const [port, setPort] = useState('value'),
    [busy, setBusy] = useState(false),
    [error, setError] = useState('');
  const [subRef, setSubRef] = useState(''),
    [subworkflow, setSubworkflow] = useState<unknown>(null);
  return (
    <Dialog title="Objects and host subworkflow boundaries" onClose={onClose}>
      <p>
        {host.host_instance_id} / {host.workspace_id}
      </p>
      <p>
        Bindings remain unresolved authoring references until the host validates the frozen draft.
        Selecting an object never runs an operation.
      </p>
      {!host.extensions?.objects && (
        <p className="notice">This host does not advertise an authorized object listing.</p>
      )}
      <button
        disabled={busy || offset === null || !host.extensions?.objects}
        onClick={async () => {
          setBusy(true);
          try {
            const p = await client.objects(offset ?? 0);
            if (p.next_offset !== null && p.next_offset <= (offset ?? 0))
              throw new Error('Object paging did not advance.');
            setObjects((old) => [...old, ...p.objects].slice(-1000));
            setOffset(p.next_offset);
          } catch (e) {
            setError(String(e));
            onHostError(e);
          } finally {
            setBusy(false);
          }
        }}
      >
        Load object summaries
      </button>
      <Field label="Input port to bind on selected node" value={port} onCommit={setPort} />
      {!node && <p>Select a node in Author to bind an object.</p>}
      {objects.map((o) => (
        <article className="group-card" key={o.object_id}>
          <h3>
            <Text>{o.object_id}</Text>
          </h3>
          <p>
            {o.type_ref} · version {o.revision}
          </p>
          <p>
            <Text>{o.summary}</Text>
          </p>
          <JsonView value={o.shape} label="Declared shape" />
          <button
            disabled={!node}
            onClick={() => {
              try {
                const key = id.parse(port);
                if (
                  node &&
                  command({
                    type: 'bindings',
                    nodeId: node.id,
                    bindings: {
                      ...node.input_bindings,
                      [key]: {
                        host_instance_id: host.host_instance_id,
                        workspace_id: host.workspace_id,
                        object_id: o.object_id,
                        resolution: 'unresolved',
                      },
                    },
                  })
                )
                  onClose();
              } catch (e) {
                setError(String(e));
              }
            }}
          >
            Bind reference to selected node
          </button>
        </article>
      ))}
      <h3>Host subworkflow</h3>
      <Field label="Subworkflow reference" value={subRef} onCommit={setSubRef} />
      <button
        disabled={busy || !host.extensions?.subworkflows || !subRef}
        onClick={async () => {
          setBusy(true);
          try {
            setSubworkflow(await client.subworkflow(id.parse(subRef)));
          } catch (e) {
            setError(String(e));
            onHostError(e);
          } finally {
            setBusy(false);
          }
        }}
      >
        Inspect host boundary
      </button>
      {!host.extensions?.subworkflows && (
        <p className="notice">
          Subworkflow navigation requires host-defined composition semantics.
        </p>
      )}
      {subworkflow !== null && (
        <JsonView value={subworkflow} label="Host subworkflow boundary mapping" />
      )}
      {error && (
        <p className="warning" role="alert">
          {error}
        </p>
      )}
    </Dialog>
  );
}
