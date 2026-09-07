import { useMemo, useState } from 'react';
import { z } from '../security/schema';
import { parseJson } from '../security/json';
import { Field, JsonView } from '../app/components';
import { integerError, rationalError, realError } from './exact';
import { reviewMatrixPaste } from './matrixPaste';

const matrix = z.strictObject({
  kind: z.literal('matrix'),
  arithmetic: z.enum(['exact', 'numerical']),
  cells: z
    .array(z.array(z.string().max(8192)).min(1).max(32))
    .min(1)
    .max(32),
});
const expression = z.strictObject({
  kind: z.literal('expression'),
  language: z.literal('mathkernel'),
  text: z.string().max(65536),
  context: z.string().max(8192),
});
export function matrixCellError(text: string, arithmetic: 'exact' | 'numerical') {
  if (arithmetic === 'numerical') return realError(text);
  const parts = text.split('/');
  return parts.length === 2 ? rationalError(parts[0]!, parts[1]!) : integerError(text);
}
export function StructuredInput({
  text,
  onChange,
}: {
  text: string;
  onChange: (text: string, transaction?: string) => void;
}) {
  const [error, setError] = useState('');
  const [paste, setPaste] = useState(''),
    [delimiter, setDelimiter] = useState<'\t' | ',' | ';'>('\t');
  const [pasteArithmetic, setPasteArithmetic] = useState<'exact' | 'numerical'>('exact');
  const review = useMemo(() => {
    if (!paste) return null;
    try {
      return { data: reviewMatrixPaste(paste, delimiter, pasteArithmetic), error: '' };
    } catch (e) {
      return { data: null, error: String(e) };
    }
  }, [paste, delimiter, pasteArithmetic]);
  let value: unknown;
  try {
    value = parseJson(text, 1048576);
  } catch {
    return null;
  }
  const m = matrix.safeParse(value),
    e = expression.safeParse(value);
  if (e.success)
    return (
      <section>
        <h3>Expression and context</h3>
        <Field
          live
          multiline
          label="Expression text"
          value={e.data.text}
          onCommit={(v, transaction) =>
            onChange(JSON.stringify({ ...e.data, text: v }), transaction)
          }
        />
        <Field
          live
          multiline
          label="Context / assumptions (draft text)"
          value={e.data.context}
          onCommit={(v, transaction) =>
            onChange(JSON.stringify({ ...e.data, context: v }), transaction)
          }
        />
        <p className="muted">
          Expression language: MathKernel. Source is displayed as text; parsing and mathematical
          meaning belong to the host.
        </p>
        <pre aria-label="Expression source preview">{e.data.text}</pre>
      </section>
    );
  if (!m.success) return null;
  const data = m.data,
    width = data.cells[0]!.length;
  const rectangular = data.cells.every((row) => row.length === width);
  const invalid = data.cells.flatMap((row, r) =>
    row.flatMap((cell, c) =>
      matrixCellError(cell, data.arithmetic)
        ? [`Row ${r + 1}, column ${c + 1}: ${matrixCellError(cell, data.arithmetic)}`]
        : [],
    ),
  );
  function resize(rows: number, cols: number) {
    if (
      !Number.isInteger(rows) ||
      !Number.isInteger(cols) ||
      rows < 1 ||
      cols < 1 ||
      rows > 32 ||
      cols > 32
    ) {
      setError(
        'Grid dimensions must be integers in 1–32. Larger inputs can use a host object binding.',
      );
      return;
    }
    if (rows < data.cells.length || cols < width) {
      setError('Shrinking would discard cells. Edit the raw draft explicitly to remove data.');
      return;
    }
    onChange(
      JSON.stringify({
        ...data,
        cells: Array.from({ length: rows }, (_, r) =>
          Array.from({ length: cols }, (_, c) => data.cells[r]?.[c] ?? ''),
        ),
      }),
    );
    setError('');
  }
  return (
    <section>
      <h3>Matrix grid</h3>
      <p>
        {data.cells.length} × {width} · {data.arithmetic} input text
      </p>
      <div className="two-columns">
        <Field
          label="Matrix rows"
          value={String(data.cells.length)}
          onCommit={(v) => resize(Number(v), width)}
        />
        <Field
          label="Matrix columns"
          value={String(width)}
          onCommit={(v) => resize(data.cells.length, Number(v))}
        />
      </div>
      <div className="table-scroll">
        <table>
          <caption>Matrix cells; row and column indices start at one</caption>
          <tbody>
            {data.cells.map((row, r) => (
              <tr key={r}>
                {row.map((cell, c) => (
                  <td key={c}>
                    <Field
                      live
                      label={`Row ${r + 1}, column ${c + 1}`}
                      value={cell}
                      onCommit={(v, transaction) => {
                        const cells = data.cells.map((a) => [...a]);
                        cells[r]![c] = v;
                        onChange(JSON.stringify({ ...data, cells }), transaction);
                      }}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">
        Exact cells accept integer or rational text. Blank cells remain blank. No conversion or
        normalization occurs.
      </p>
      <details>
        <summary>Paste a table with review</summary>
        <p>
          Applying replaces the entire grid in one undoable edit. Text is retained exactly. Formulas
          and quoted CSV syntax are unsupported.
        </p>
        <label className="field">
          Delimiter
          <select
            value={delimiter}
            onChange={(e) => setDelimiter(e.target.value as typeof delimiter)}
          >
            <option value={'\t'}>Tab</option>
            <option value=",">Comma</option>
            <option value=";">Semicolon</option>
          </select>
        </label>
        <label className="field">
          Pasted arithmetic
          <select
            value={pasteArithmetic}
            onChange={(e) => setPasteArithmetic(e.target.value as typeof pasteArithmetic)}
          >
            <option value="exact">Exact integers / rationals</option>
            <option value="numerical">Numerical — decimal point</option>
          </select>
        </label>
        <label className="field">
          Pasted table text
          <textarea
            value={paste}
            maxLength={1024 * 1024}
            onChange={(e) => setPaste(e.target.value)}
          />
        </label>
        {review?.error && <p className="warning">{review.error}</p>}
        {review?.data && (
          <>
            <p>
              {review.data.rows} rows × {review.data.columns} columns · decimal point ·{' '}
              {review.data.rejected.length} rejected cells
            </p>
            <JsonView
              value={
                review.data.rejected.length ? review.data.rejected.slice(0, 32) : review.data.cells
              }
            />
            <button
              disabled={review.data.rejected.length > 0}
              onClick={() => {
                onChange(
                  JSON.stringify({
                    kind: 'matrix',
                    arithmetic: review.data!.arithmetic,
                    cells: review.data!.cells,
                  }),
                );
                setPaste('');
              }}
            >
              Replace grid with reviewed table
            </button>
          </>
        )}
      </details>
      {(!rectangular || error || invalid.length > 0) && (
        <p role="status" className="warning">
          {error || (!rectangular ? 'Rows have different lengths.' : invalid.slice(0, 4).join(' '))}
        </p>
      )}
    </section>
  );
}

/** Interpret only a bounded literal schema subset; never execute patterns, refs, or plugins. */
export function ParameterControl({
  name,
  schema,
  value,
  onChange,
}: {
  name: string;
  schema: unknown;
  value: string | undefined;
  onChange: (text: string, encoding: 'text' | 'json_text', transaction?: string) => void;
}) {
  const s =
    schema && typeof schema === 'object' && !Array.isArray(schema)
      ? (schema as Record<string, unknown>)
      : {};
  if (s.type === 'boolean')
    return (
      <label className="field">
        {name}
        <select value={value ?? ''} onChange={(e) => onChange(e.target.value, 'json_text')}>
          <option value="" disabled>
            Omitted
          </option>
          <option value="true">True</option>
          <option value="false">False</option>
          {value !== undefined && !['true', 'false'].includes(value) && (
            <option value={value}>Invalid draft: {value.slice(0, 80)}</option>
          )}
        </select>
      </label>
    );
  const choices =
    Array.isArray(s.enum) &&
    s.enum.length <= 100 &&
    s.enum.every(
      (v) =>
        typeof v === 'string' ||
        typeof v === 'boolean' ||
        (typeof v === 'number' && Number.isSafeInteger(v)),
    )
      ? s.enum
      : null;
  if (choices)
    return (
      <label className="field">
        {name}
        <select value={value ?? ''} onChange={(e) => onChange(e.target.value, 'json_text')}>
          <option value="" disabled>
            Omitted
          </option>
          {choices.map((v) => (
            <option key={JSON.stringify(v)} value={JSON.stringify(v)}>
              {String(v)}
            </option>
          ))}
          {value !== undefined && !choices.some((v) => JSON.stringify(v) === value) && (
            <option value={value}>Unresolved draft: {value.slice(0, 80)}</option>
          )}
        </select>
      </label>
    );
  return (
    <>
      <Field
        live
        multiline
        label={name}
        value={value ?? ''}
        onCommit={(v, transaction) => onChange(v, 'text', transaction)}
      />
      {Object.keys(s).length > 0 && (
        <details>
          <summary>Declared constraints</summary>
          <JsonView value={s} />
          <p className="muted">
            Host validation required. Constraints do not transform this text or load external
            schemas.
          </p>
        </details>
      )}
    </>
  );
}
