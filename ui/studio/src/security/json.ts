/** Bounded, duplicate-aware JSON parser. Never evaluates code or resolves URLs. */
export const DOCUMENT_BYTES = 10 * 1024 * 1024;
export const CONTROL_BYTES = 2 * 1024 * 1024;
const forbidden = new Set(['__proto__', 'constructor', 'prototype']);

export function parseJson(text: string, maxBytes = DOCUMENT_BYTES): unknown {
  if (text.length > maxBytes || new TextEncoder().encode(text).length > maxBytes)
    throw new Error('JSON exceeds the byte limit.');
  let i = 0,
    values = 0;
  const whitespace = () => {
    while (i < text.length && /[\x20\t\r\n]/.test(text[i]!)) i++;
  };
  const fail = (message: string): never => {
    throw new Error(`${message} at character ${i}.`);
  };
  function string(): string {
    const start = i++;
    while (i < text.length) {
      const c = text[i++];
      if (c === '"') {
        try {
          return JSON.parse(text.slice(start, i)) as string;
        } catch {
          return fail('Invalid JSON string');
        }
      }
      if (c === '\\') i++;
    }
    return fail('Unterminated string');
  }
  function value(depth: number): unknown {
    whitespace();
    if (depth > 32 || ++values > 250000) fail('JSON complexity limit exceeded');
    const c = text[i];
    if (c === '"') return string();
    if (c === '{') {
      i++;
      whitespace();
      const out: Record<string, unknown> = Object.create(null);
      if (text[i] === '}') {
        i++;
        return out;
      }
      while (i < text.length) {
        if (text[i] !== '"') fail('Expected an object key');
        const key = string();
        if (forbidden.has(key)) fail('Prohibited prototype-related property');
        if (Object.hasOwn(out, key)) fail('Duplicate JSON key');
        whitespace();
        if (text[i++] !== ':') fail('Expected colon');
        out[key] = value(depth + 1);
        whitespace();
        if (text[i] === '}') {
          i++;
          return out;
        }
        if (text[i++] !== ',') fail('Expected comma');
        whitespace();
      }
      return fail('Unterminated object');
    }
    if (c === '[') {
      i++;
      whitespace();
      const out: unknown[] = [];
      if (text[i] === ']') {
        i++;
        return out;
      }
      while (i < text.length) {
        out.push(value(depth + 1));
        whitespace();
        if (text[i] === ']') {
          i++;
          return out;
        }
        if (text[i++] !== ',') fail('Expected comma');
      }
      return fail('Unterminated array');
    }
    for (const [token, result] of [
      ['true', true],
      ['false', false],
      ['null', null],
    ] as const) {
      if (text.startsWith(token, i)) {
        i += token.length;
        return result;
      }
    }
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(text.slice(i));
    if (!match) return fail('Expected a JSON value');
    i += match[0].length;
    const result = Number(match[0]);
    if (!Number.isFinite(result) || (Number.isInteger(result) && !Number.isSafeInteger(result)))
      fail('Unsafe JSON number; mathematical integers must use strings');
    return result;
  }
  const result = value(0);
  whitespace();
  if (i !== text.length) fail('Trailing JSON content');
  return result;
}

export function displayText(text: string): string {
  // Make terminal escapes, bidi controls and invisible identifier characters visible.
  return text.replace(
    /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u202a-\u202e\u2066-\u2069]/g,
    (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`,
  );
}
