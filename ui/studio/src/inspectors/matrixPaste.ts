import { integerError, rationalError, realError } from './exact';

/** A literal table parser. Formulas, decimal conventions and exact text are never evaluated. */
export function reviewMatrixPaste(
  text: string,
  delimiter: '\t' | ',' | ';',
  arithmetic: 'exact' | 'numerical',
) {
  if (new TextEncoder().encode(text).length > 1024 * 1024) throw new Error('Paste exceeds 1 MiB.');
  const lines = text.replace(/\r\n?/g, '\n').replace(/\n$/, '').split('\n');
  const cells = lines.map((line) => line.split(delimiter));
  const width = cells[0]?.length ?? 0;
  if (!width || width > 32 || cells.length > 32)
    throw new Error('Paste supports 1–32 rows and columns.');
  if (cells.some((row) => row.length !== width))
    throw new Error('Rows have different column counts. Select the actual delimiter.');
  const rejected: { row: number; column: number; text: string; reason: string }[] = [];
  cells.forEach((row, r) =>
    row.forEach((cell, c) => {
      if (cell.length > 8192) throw new Error('A cell exceeds 8192 characters.');
      const parts = cell.split('/');
      const reason =
        arithmetic === 'numerical'
          ? realError(cell)
          : parts.length === 2
            ? rationalError(parts[0]!, parts[1]!)
            : integerError(cell);
      if (reason) rejected.push({ row: r + 1, column: c + 1, text: cell, reason });
    }),
  );
  return {
    cells,
    rows: cells.length,
    columns: width,
    delimiter,
    arithmetic,
    decimal_convention: 'point',
    rejected,
  };
}
