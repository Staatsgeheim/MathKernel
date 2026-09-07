import { useMemo, useState } from 'react';
import { Text } from '../app/components';
import { downloadJson } from '../editor/CompositionPanel';

/** Inspect literal row arrays only. Values retain their original JSON representation. */
export function sourceTables(value: unknown) {
  const entries = Array.isArray(value)
    ? [['value', value] as const]
    : value && typeof value === 'object'
      ? Object.entries(value).slice(0, 100)
      : [];
  return entries.filter(
    (entry): entry is [string, unknown[][]] =>
      Array.isArray(entry[1]) &&
      entry[1].length > 0 &&
      entry[1].length <= 5000 &&
      entry[1].every((row) => Array.isArray(row) && row.length <= 100),
  );
}
export function canonicalCell(value: unknown): string {
  return typeof value === 'string' ? value : JSON.stringify(value);
}
export function SourceTables({ value }: { value: unknown }) {
  const tables = useMemo(() => sourceTables(value), [value]);
  return (
    <>
      {tables.map(([name, rows]) => (
        <SourceTable key={name} name={name} rows={rows} />
      ))}
    </>
  );
}
function SourceTable({ name, rows }: { name: string; rows: unknown[][] }) {
  const [page, setPage] = useState(0),
    [message, setMessage] = useState('');
  const start = page * 50,
    current = rows.slice(start, start + 50);
  return (
    <section>
      <h3>
        Source table: <Text>{name}</Text>
      </h3>
      <p>
        Rows {start + 1}–{Math.min(start + 50, rows.length)} of {rows.length}. Source order; no
        sorting, rounding or recomputation.
      </p>
      <div className="table-scroll">
        <table>
          <caption>Literal source cells; row and column indices start at one</caption>
          <thead>
            <tr>
              <th>Row</th>
              {Array.from({ length: Math.max(...current.map((r) => r.length)) }, (_, i) => (
                <th key={i}>{i + 1}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {current.map((row, r) => (
              <tr key={start + r}>
                <th>{start + r + 1}</th>
                {row.map((cell, c) => (
                  <td key={c}>
                    <Text>{canonicalCell(cell)}</Text>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="toolbar">
        <button disabled={!page} onClick={() => setPage(page - 1)}>
          Previous table page
        </button>
        <button disabled={start + 50 >= rows.length} onClick={() => setPage(page + 1)}>
          Next table page
        </button>
        <button
          onClick={async () => {
            try {
              if (!navigator.clipboard)
                throw new Error('Clipboard unavailable; export the source table instead.');
              await navigator.clipboard.writeText(JSON.stringify(current));
              setMessage('Copied this page as exact source JSON.');
            } catch (e) {
              setMessage(String(e));
            }
          }}
        >
          Copy this page as source JSON
        </button>
        <button onClick={() => downloadJson(rows, `${name}-source-table.json`)}>
          Export complete source table
        </button>
      </div>
      {message && (
        <p role="status">
          <Text>{message}</Text>
        </p>
      )}
    </section>
  );
}
