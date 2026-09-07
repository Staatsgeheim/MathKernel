import { useState } from 'react';
import { descriptor, type Command } from '../editor/commands';
import type { DraftNode, Position, StudioDocument } from '../editor/document';
import type { Operation } from '../host/contracts';
import { scalarDraft, integerError, rationalError, realError, type ScalarDraft } from './exact';
import { Field, JsonView, Text } from '../app/components';
import { parseJson } from '../security/json';
import { StructuredInput, ParameterControl } from './StructuredInput';

export function NodeInspector({
  node,
  position,
  catalog,
  document,
  command,
}: {
  node: DraftNode;
  position: Position;
  catalog: readonly Operation[];
  document: StudioDocument;
  command: (c: Command) => void;
}) {
  const op = descriptor(node, catalog);
  const scalar =
    node.kind === 'object_binding' && node.operation_ref === null
      ? scalarDraft(node.parameter_drafts.value?.text ?? '')
      : null;
  const [parameterName, setParameterName] = useState('');
  const [localError, setLocalError] = useState('');
  const setValue = (value: ScalarDraft, transaction?: string) =>
    command({
      type: 'parameters',
      transaction,
      nodeId: node.id,
      drafts: {
        ...node.parameter_drafts,
        value: { encoding: 'json_text', text: JSON.stringify(value, null, 2) },
      },
    });
  const properties = op?.parameter_schema.properties;
  const keys = new Set([
    ...Object.keys(node.parameter_drafts),
    ...(properties && typeof properties === 'object' ? Object.keys(properties) : []),
  ]);
  return (
    <div className="inspector-content">
      <p className="eyebrow">Selected node</p>
      <h2>
        <Text>{node.label}</Text>
      </h2>
      <code className="wrap">
        <Text>{node.operation_ref ?? 'Local input draft'}</Text>
      </code>
      <p className="notice">
        {op
          ? op.coverage
          : 'Unresolved authoring intent. No host execution or validation has occurred.'}
      </p>
      <Field
        label="Node label"
        value={node.label}
        onCommit={(label) => command({ type: 'label', nodeId: node.id, label })}
      />
      {node.kind === 'object_binding' && node.parameter_drafts.value && (
        <StructuredInput
          text={node.parameter_drafts.value.text}
          onChange={(text, transaction) =>
            command({
              type: 'parameters',
              nodeId: node.id,
              transaction,
              drafts: { ...node.parameter_drafts, value: { encoding: 'json_text', text } },
            })
          }
        />
      )}
      {scalar && (
        <section>
          <h3>{scalar.kind === 'real' ? 'Numerical input' : 'Exact input'}</h3>
          {scalar.kind === 'rational' ? (
            <>
              <Field
                live
                label="Numerator (exact text)"
                value={scalar.numerator}
                onCommit={(numerator, transaction) =>
                  setValue({ ...scalar, numerator }, transaction)
                }
              />
              <Field
                live
                label="Denominator (exact text)"
                value={scalar.denominator}
                onCommit={(denominator, transaction) =>
                  setValue({ ...scalar, denominator }, transaction)
                }
              />
              <p role="status" className="warning">
                {rationalError(scalar.numerator, scalar.denominator)}
              </p>
            </>
          ) : (
            <>
              <Field
                live
                label={
                  scalar.kind === 'integer' ? 'Integer (exact text)' : 'Decimal (approximate text)'
                }
                value={scalar.value}
                onCommit={(value, transaction) => setValue({ ...scalar, value }, transaction)}
              />
              <p role="status" className="warning">
                {scalar.kind === 'integer' ? integerError(scalar.value) : realError(scalar.value)}
              </p>
            </>
          )}
          {scalar.kind === 'real' && (
            <Field
              label="Declared precision (bits; blank = unspecified)"
              value={scalar.precision === null ? '' : String(scalar.precision)}
              onCommit={(value) => {
                if (value === '') {
                  setValue({ ...scalar, precision: null });
                  setLocalError('');
                } else if (/^\d{1,7}$/.test(value) && Number(value) > 0) {
                  setValue({ ...scalar, precision: Number(value) });
                  setLocalError('');
                } else
                  setLocalError(
                    'Precision must be a positive integer. The draft value was not changed.',
                  );
              }}
            />
          )}
          <p className="muted">
            Input text is preserved verbatim. Local format checks do not establish mathematical
            evidence.
          </p>
        </section>
      )}
      <section>
        <h3>Parameter drafts</h3>
        <p className="muted">
          Text updates the draft as you type; undo groups each field edit. Invalid text can be saved
          for recovery. Omitted values stay omitted.
        </p>
        {[...keys].map((key) => (
          <div key={key} className="parameter-field">
            <ParameterControl
              name={`${key}${Object.hasOwn(node.parameter_drafts, key) ? '' : ' (omitted)'}`}
              schema={
                properties && typeof properties === 'object'
                  ? (properties as Record<string, unknown>)[key]
                  : undefined
              }
              value={node.parameter_drafts[key]?.text}
              onChange={(text, encoding, transaction) =>
                command({
                  type: 'parameters',
                  transaction,
                  nodeId: node.id,
                  drafts: {
                    ...node.parameter_drafts,
                    [key]: { encoding: node.parameter_drafts[key]?.encoding ?? encoding, text },
                  },
                })
              }
            />
            {node.parameter_drafts[key] && (
              <button
                onClick={() => {
                  const drafts = { ...node.parameter_drafts };
                  delete drafts[key];
                  command({ type: 'parameters', nodeId: node.id, drafts });
                }}
              >
                Omit {key}
              </button>
            )}
            {node.parameter_drafts[key]?.encoding === 'json_text' && (
              <JsonFormat text={node.parameter_drafts[key]!.text} />
            )}
          </div>
        ))}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (
              !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(parameterName) ||
              ['constructor', 'prototype', '__proto__'].includes(parameterName)
            ) {
              setLocalError('Use a valid parameter identifier.');
              return;
            }
            if (Object.hasOwn(node.parameter_drafts, parameterName)) {
              setLocalError('Parameter already exists.');
              return;
            }
            command({
              type: 'parameters',
              nodeId: node.id,
              drafts: { ...node.parameter_drafts, [parameterName]: { encoding: 'text', text: '' } },
            });
            setParameterName('');
            setLocalError('');
          }}
        >
          <label className="field">
            Additional draft parameter
            <input
              value={parameterName}
              onChange={(e) => setParameterName(e.target.value)}
              maxLength={128}
            />
          </label>
          <button>Add parameter</button>
        </form>
      </section>
      <details>
        <summary>Position and identity</summary>
        <div className="two-columns">
          <Field
            label="X position"
            value={String(position.x)}
            onCommit={(x) => {
              if (!/^[+-]?\d+(\.\d+)?$/.test(x)) {
                setLocalError('Enter a numeric position.');
                return;
              }
              command({ type: 'move', positions: { [node.id]: { ...position, x: Number(x) } } });
            }}
          />
          <Field
            label="Y position"
            value={String(position.y)}
            onCommit={(y) => {
              if (!/^[+-]?\d+(\.\d+)?$/.test(y)) {
                setLocalError('Enter a numeric position.');
                return;
              }
              command({ type: 'move', positions: { [node.id]: { ...position, y: Number(y) } } });
            }}
          />
        </div>
        <p className="muted wrap">Node: {node.id}</p>
        <p className="muted wrap">Schema: {node.schema_digest ?? 'unresolved'}</p>
      </details>
      <details>
        <summary>Ports, bindings and desired outputs</summary>
        {op && (
          <>
            <JsonView value={{ inputs: op.input_ports, outputs: op.output_ports }} />
            {op.output_ports.map((p) => (
              <label className="check" key={p.id}>
                <input
                  type="checkbox"
                  checked={document.authoring.desired_outputs.some(
                    (o) => o.node_id === node.id && o.port_id === p.id,
                  )}
                  onChange={(e) =>
                    command({
                      type: 'output',
                      nodeId: node.id,
                      portId: p.id,
                      selected: e.target.checked,
                    })
                  }
                />
                Request output {p.label}
              </label>
            ))}
          </>
        )}
        <JsonView value={node.input_bindings} label="Unresolved object bindings" />
      </details>
      {op && (
        <details>
          <summary>Operation documentation</summary>
          <p>
            <Text>{op.description}</Text>
          </p>
          <p>Engine declarations: {op.engines.join(', ') || 'Not reported'}</p>
          <p>
            Potential trust levels: {op.trust_levels.join(', ') || 'Not reported'}. These are
            capabilities, not results.
          </p>
          <JsonView value={op.parameter_schema} label="Declared parameter schema" />
        </details>
      )}
      {localError && (
        <p role="alert" className="warning">
          {localError}
        </p>
      )}
    </div>
  );
}
function JsonFormat({ text }: { text: string }) {
  try {
    parseJson(text, 1048576);
    return <p className="muted">JSON format parsed; host validation required.</p>;
  } catch (e) {
    return (
      <p className="warning">
        {e instanceof Error ? e.message : 'Invalid JSON'} Your text is retained.
      </p>
    );
  }
}
