import { useState } from 'react';
import { Dialog, Field, JsonView, Text } from '../app/components';
import type { HostCommandService } from './client';
import type { Handshake } from './contracts';
import type { DraftNode } from '../editor/document';
import { id } from '../editor/document';
import type { z } from '../security/schema';
import { subworkflowSchema } from './workflow';
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
    [subworkflow, setSubworkflow] = useState<z.infer<typeof subworkflowSchema> | null>(null);
  const [previousBoundary, setPreviousBoundary] = useState<z.infer<
    typeof subworkflowSchema
  > | null>(null);
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
            const next = await client.subworkflow(id.parse(subRef));
            if (
              subworkflow?.reference === next.reference &&
              subworkflow.revision === next.revision &&
              subworkflow.digest !== next.digest
            )
              throw new Error(
                'Host changed a subworkflow digest at the same revision. Reconnect before trusting this boundary.',
              );
            setPreviousBoundary(subworkflow);
            setSubworkflow(next);
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
        <section>
          <h4>
            <Text>{subworkflow.title}</Text> · revision {subworkflow.revision}
          </h4>
          <p>
            {subworkflow.read_only
              ? 'Read-only host boundary'
              : 'Host permits editing; this client exposes inspection only'}
            . This does not inline or execute its contents.
          </p>
          <p className="wrap">Digest: {subworkflow.digest}</p>
          <table>
            <caption>External to internal port mapping</caption>
            <thead>
              <tr>
                <th>External port</th>
                <th>Internal node</th>
                <th>Internal port</th>
              </tr>
            </thead>
            <tbody>
              {subworkflow.boundary.map((b, i) => (
                <tr key={i}>
                  <td>
                    <Text>{b.external_port}</Text>
                  </td>
                  <td>
                    <Text>{b.internal_node}</Text>
                  </td>
                  <td>
                    <Text>{b.internal_port}</Text>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <JsonView value={subworkflow.required_capabilities} label="Required host capabilities" />
          <p>
            <Text>{subworkflow.description}</Text>
          </p>
          {previousBoundary && (
            <details>
              <summary>
                Previously inspected boundary: {previousBoundary.reference} ·{' '}
                {previousBoundary.revision}
              </summary>
              <p>
                {previousBoundary.digest === subworkflow.digest
                  ? 'Digest unchanged.'
                  : 'Boundary digest changed. Existing drafts are not rebound automatically.'}
              </p>
              <JsonView value={previousBoundary} label="Previous immutable boundary" />
            </details>
          )}
        </section>
      )}
      {error && (
        <p className="warning" role="alert">
          {error}
        </p>
      )}
    </Dialog>
  );
}
