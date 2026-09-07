import { memo, useCallback, useEffect, useMemo, useState } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  applyNodeChanges,
  MarkerType,
  type Node,
  type NodeProps,
  type NodeChange,
  type Connection,
  type Edge,
  type ReactFlowInstance,
} from '@xyflow/react';
import type { StudioDocument } from './document';
import { newId } from './document';
import { descriptor, compatibility, type Command } from './commands';
import type { Operation, Port } from '../host/contracts';
import { Text } from '../app/components';
import '@xyflow/react/dist/style.css';

type NodeData = { label: string; operation: string; state: string; ports: Port[]; summary: string };
type FlowNode = Node<NodeData, 'studio'>;
const StudioNode = memo(function StudioNode({ data, selected }: NodeProps<FlowNode>) {
  return (
    <article className={`studio-node ${selected ? 'is-selected' : ''}`}>
      <div className="node-handle">
        <span className="node-symbol" aria-hidden="true">
          ƒ
        </span>
        <strong>
          <Text>{data.label}</Text>
        </strong>
      </div>
      <div className="node-body">
        <code>
          <Text>{data.operation}</Text>
        </code>
        <p>
          <Text>{data.summary}</Text>
        </p>
        <span className="badge">
          <Text>{data.state}</Text>
        </span>
      </div>
      {data.ports.map((p, i) => (
        <Handle
          key={`${p.direction}-${p.id}`}
          type={p.direction === 'input' ? 'target' : 'source'}
          position={p.direction === 'input' ? Position.Left : Position.Right}
          id={p.id}
          style={{ top: 62 + i * 24 }}
          aria-label={`${p.direction} ${p.label}, ${p.type_ref ?? 'type unresolved'}`}
        />
      ))}
    </article>
  );
});
const nodeTypes = { studio: StudioNode };
export function Canvas({
  document,
  catalog,
  selected,
  onSelect,
  command,
  connect,
}: {
  document: StudioDocument;
  catalog: readonly Operation[];
  selected: string[];
  onSelect: (ids: string[]) => void;
  command: (c: Command) => void;
  connect: (edge: StudioDocument['authoring']['edges'][number], oldId?: string) => void;
}) {
  const model = useMemo<FlowNode[]>(
    () =>
      document.authoring.nodes.map((n, i) => {
        const op = descriptor(n, catalog);
        const ports = [...(op?.input_ports ?? []), ...(op?.output_ports ?? [])];
        for (const e of document.authoring.edges) {
          for (const [node, port, direction] of [
            [e.source_node, e.source_port, 'output'],
            [e.target_node, e.target_port, 'input'],
          ] as const)
            if (node === n.id && !ports.some((p) => p.id === port && p.direction === direction))
              ports.push({
                id: port,
                direction,
                label: `${port} (unresolved)`,
                type_ref: null,
                cardinality: 'unknown',
                required: null,
                shape: null,
              });
        }
        return {
          id: n.id,
          type: 'studio',
          position: document.presentation.node_positions[n.id] ?? { x: 40 + i * 300, y: 80 },
          dragHandle: '.node-handle',
          selected: selected.includes(n.id),
          data: {
            label: n.label,
            operation: n.operation_ref ?? 'Local input draft',
            state: op
              ? `${op.composition} · not run`
              : n.operation_ref
                ? 'Unresolved · not run'
                : 'Draft input · not run',
            ports,
            summary:
              Object.values(n.parameter_drafts)[0]?.text.slice(0, 85) ?? 'Select to configure',
          },
        };
      }),
    [document, catalog, selected],
  );
  const [nodes, setNodes] = useState(model);
  const [flow, setFlow] = useState<ReactFlowInstance<FlowNode, Edge> | null>(null);
  useEffect(() => setNodes(model), [model]);
  const edges = useMemo<Edge[]>(
    () =>
      document.authoring.edges.map((e) => ({
        id: e.id,
        source: e.source_node,
        target: e.target_node,
        sourceHandle: e.source_port,
        targetHandle: e.target_port,
        markerEnd: { type: MarkerType.ArrowClosed },
        label: compatibility(document, catalog, e).state.replace('_', ' '),
        reconnectable: true,
      })),
    [document, catalog],
  );
  const onNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => {
      setNodes((current) =>
        applyNodeChanges(
          changes.filter((c) => c.type !== 'remove'),
          current,
        ),
      );
      const moves = changes.filter(
        (c): c is Extract<NodeChange<FlowNode>, { type: 'position' }> =>
          c.type === 'position' && !!c.position && c.dragging === false,
      );
      if (moves.length)
        command({
          type: 'move',
          positions: Object.fromEntries(
            moves.map((c) => [c.id, c.type === 'position' ? c.position! : { x: 0, y: 0 }]),
          ),
        });
    },
    [command],
  );
  const wire = (c: Connection, oldId?: string) => {
    if (c.source && c.target && c.sourceHandle && c.targetHandle)
      connect(
        {
          id: oldId ?? newId(),
          source_node: c.source,
          source_port: c.sourceHandle,
          target_node: c.target,
          target_port: c.targetHandle,
        },
        oldId,
      );
  };
  return (
    <div className="canvas" aria-label="Authoring canvas; an equivalent outline view is available">
      <ReactFlow<FlowNode, Edge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={setFlow}
        onNodesChange={onNodesChange}
        onSelectionChange={({ nodes }) => {
          const ids = nodes.map((n) => n.id);
          if (ids.join() !== selected.join()) onSelect(ids);
        }}
        onConnect={(c) => wire(c)}
        onReconnect={(old, c) => wire(c, old.id)}
        onNodeDragStop={(_, __, moved) =>
          command({
            type: 'move',
            positions: Object.fromEntries(moved.map((n) => [n.id, n.position])),
          })
        }
        defaultViewport={document.presentation.viewport}
        onMoveEnd={(_, viewport) => command({ type: 'viewport', viewport })}
        deleteKeyCode={null}
        selectionOnDrag
        minZoom={0.05}
        maxZoom={2}
        colorMode="system"
        onlyRenderVisibleElements
      >
        <Background gap={24} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
      {document.authoring.nodes.length === 0 && (
        <div className="canvas-empty">
          <span className="empty-symbol" aria-hidden="true">
            ƒ
          </span>
          <h2>Your next experiment starts here</h2>
          <p>
            Add an exact input or browse the operation catalog.
            <br />
            Your graph stays an editable draft.
          </p>
        </div>
      )}
      <button className="fit-button" onClick={() => flow?.fitView({ padding: 0.15, duration: 0 })}>
        Fit graph
      </button>
    </div>
  );
}
