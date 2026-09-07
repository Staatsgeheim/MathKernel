import { expect, it, vi } from 'vitest';
import { reviewMatrixPaste } from '../src/inspectors/matrixPaste';
import { canonicalCell, sourceTables } from '../src/viewers/SourceTable';
import { randomId } from '../src/security/identity';

it('table paste preserves exact cells and discloses dimensions without normalization', () => {
  const v = reviewMatrixPaste('9007199254740993123456789\t1/3\r\n-2\t0\r\n', '\t', 'exact');
  expect(v.rows).toBe(2);
  expect(v.columns).toBe(2);
  expect(v.rejected).toEqual([]);
  expect(v.cells[0]).toEqual(['9007199254740993123456789', '1/3']);
});
it('paste review rejects formulas, zero denominators and ambiguous decimal commas', () => {
  expect(reviewMatrixPaste('=1+1\t1/0', '\t', 'exact').rejected).toHaveLength(2);
  expect(reviewMatrixPaste('1,5;2,5', ';', 'numerical').rejected).toHaveLength(2);
  expect(() => reviewMatrixPaste('1\t2\n3', '\t', 'exact')).toThrow('different column');
  expect(() => reviewMatrixPaste(Array(33).fill('1').join('\n'), '\t', 'exact')).toThrow('1–32');
});
it('a failed paste does not silently drop blanks, quoted cells or extra precision', () => {
  const v = reviewMatrixPaste('"1"\t\t01', '\t', 'exact');
  expect(v.cells).toEqual([['"1"', '', '01']]);
  expect(v.rejected.length).toBeGreaterThan(0);
});
it('source table inspection retains large integers and rational object structure', () => {
  const exact = { kind: 'rational', numerator: '12345678901234567890123456789', denominator: '7' };
  expect(canonicalCell(exact)).toBe(JSON.stringify(exact));
  expect(canonicalCell('9007199254740993123456789')).toBe('9007199254740993123456789');
  const tables = sourceTables({ rows: [[exact]], count: 1 });
  expect(tables).toEqual([['rows', [[exact]]]]);
  expect(sourceTables({ rows: Array.from({ length: 5001 }, () => [0]) })).toEqual([]);
});
it('opaque IDs remain cryptographically random when randomUUID is unavailable', () => {
  const original = crypto;
  vi.stubGlobal('crypto', { getRandomValues: original.getRandomValues.bind(original) });
  try {
    const ids = Array.from({ length: 32 }, randomId);
    expect(new Set(ids).size).toBe(32);
    expect(
      ids.every((id) =>
        /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/.test(id),
      ),
    ).toBe(true);
  } finally {
    vi.unstubAllGlobals();
  }
});
