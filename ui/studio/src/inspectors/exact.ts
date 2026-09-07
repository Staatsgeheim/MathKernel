import { parseJson } from '../security/json';
const integer = /^[+-]?\d+$/;
export function integerError(value: string): string | null {
  if (!value) return 'Blank is not zero. Enter a signed decimal integer.';
  return integer.test(value)
    ? null
    : 'Use decimal integer text without separators; exact values are never rounded.';
}
export function rationalError(numerator: string, denominator: string): string | null {
  return (
    integerError(numerator) ||
    integerError(denominator) ||
    (/^[+-]?0+$/.test(denominator) ? 'Denominator must not be zero.' : null)
  );
}
export function realError(value: string): string | null {
  if (value.includes(',')) return 'Use a decimal point. Decimal-comma input is not enabled.';
  return /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)
    ? null
    : 'Enter finite decimal text; blank, NaN and infinity are not supported.';
}
export type ScalarDraft =
  | { kind: 'integer'; value: string }
  | { kind: 'rational'; numerator: string; denominator: string }
  | { kind: 'real'; value: string; precision: number | null };
export function scalarDraft(text: string): ScalarDraft | null {
  try {
    const v = parseJson(text, 1048576) as Record<string, unknown>;
    if (!v || typeof v !== 'object') return null;
    if (v.kind === 'integer' && typeof v.value === 'string' && Object.keys(v).length === 2)
      return { kind: v.kind, value: v.value };
    if (
      v.kind === 'rational' &&
      typeof v.numerator === 'string' &&
      typeof v.denominator === 'string' &&
      Object.keys(v).length === 3
    )
      return { kind: v.kind, numerator: v.numerator, denominator: v.denominator };
    if (
      v.kind === 'real' &&
      typeof v.value === 'string' &&
      (v.precision === null ||
        (typeof v.precision === 'number' &&
          Number.isSafeInteger(v.precision) &&
          v.precision > 0)) &&
      Object.keys(v).length === 3
    )
      return { kind: v.kind, value: v.value, precision: v.precision as number | null };
  } catch {
    /* Invalid buffers remain editable as text. */
  }
  return null;
}
