import { z } from '../security/schema';
import { operationSchema } from '../host/contracts';
import { parseJson } from '../security/json';
import { id } from './document';
const cacheSchema = z.strictObject({
  host: id,
  workspace: id,
  revision: id,
  savedAt: z.iso.datetime(),
  entries: operationSchema.array().max(20000),
});
export function saveCatalog(value: z.infer<typeof cacheSchema>) {
  const text = JSON.stringify(cacheSchema.parse(value));
  if (text.length > 2 * 1024 * 1024)
    throw new Error('Catalog is too large for the optional browser cache.');
  localStorage.setItem('mk-studio-catalog-1', text);
}
export function loadCatalog() {
  const text = localStorage.getItem('mk-studio-catalog-1');
  return text ? cacheSchema.parse(parseJson(text, 2 * 1024 * 1024)) : null;
}
export function clearCatalog() {
  localStorage.removeItem('mk-studio-catalog-1');
}
