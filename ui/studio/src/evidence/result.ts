import { mathResultSchema, type ResultObservation } from '../host/contracts';
import { parseJson } from '../security/json';
export function receipt(
  value: unknown,
): { resource_id: string; size_bytes: number; sha256?: string } | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  const data = (v.data && typeof v.data === 'object' ? v.data : v) as Record<string, unknown>;
  if (
    data.truncated === true &&
    typeof data.resource_id === 'string' &&
    /^[A-Za-z0-9_.:-]{1,128}$/.test(data.resource_id) &&
    Number.isSafeInteger(data.size_bytes) &&
    (data.size_bytes as number) >= 0
  )
    return {
      resource_id: data.resource_id,
      size_bytes: data.size_bytes as number,
      sha256: typeof data.sha256 === 'string' ? data.sha256 : undefined,
    };
  return null;
}
export function admittedResult(observation: ResultObservation) {
  if (observation.admission !== 'host' || receipt(observation.result)) return null;
  return mathResultSchema.parse(observation.result);
}
export async function verifyPages(
  content: string,
  totalBytes: number,
  digest: string,
): Promise<unknown> {
  const bytes = new TextEncoder().encode(content);
  if (bytes.length !== totalBytes) throw new Error('Partial result loaded.');
  const actual = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((v) => v.toString(16).padStart(2, '0'))
    .join('');
  if (actual !== digest) throw new Error('Result integrity mismatch.');
  return parseJson(content, 2 * 1024 * 1024);
}
