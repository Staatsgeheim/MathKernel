import { readFile, readdir, writeFile, mkdir, copyFile, rm } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
const root = path.resolve('host/src/mathkernel_studio/assets');
const lock = JSON.parse(await readFile('package-lock.json', 'utf8'));
const packages = Object.entries(lock.packages)
  .filter(([name]) => name)
  .map(([name, info]) => ({
    name: name.replace(/^node_modules\//, ''),
    version: info.version,
    license: info.license ?? 'SEE PACKAGE NOTICE',
    integrity: info.integrity,
    development: !!info.dev,
  }));
await mkdir(root, { recursive: true });
const viewerScript = await readFile(path.join(root, 'viewer', 'viewer.js'), 'utf8');
const viewerStyle = await readFile(path.join(root, 'viewer.css'), 'utf8');
let viewerHtml = await readFile(path.join(root, 'viewer.html'), 'utf8');
viewerHtml = viewerHtml
  .replace(
    '<link rel="stylesheet" href="/studio/viewer.css">',
    () => `<style>${viewerStyle}</style>`,
  )
  .replace(
    '<script defer src="/studio/viewer/viewer.js"></script>',
    () => `<script>${viewerScript}</script>`,
  );
if (
  viewerHtml.includes('<link rel="stylesheet" href="/studio/viewer.css">') ||
  viewerHtml.includes('<script defer src="/studio/viewer/viewer.js"></script>')
)
  throw new Error('Viewer assets were not inlined');
await writeFile(path.join(root, 'viewer.html'), viewerHtml);
const viewerHash = createHash('sha256').update(viewerScript).digest('base64');
await writeFile(
  path.join(root, 'viewer-csp.txt'),
  `default-src 'none'; script-src 'sha256-${viewerHash}'; style-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'self'; form-action 'none'`,
);
await rm(path.join(root, 'viewer'), { recursive: true, force: true });
await rm(path.join(root, 'viewer.css'), { force: true });
const notices = [];
for (const [dir, info] of Object.entries(lock.packages)) {
  if (!dir) continue;
  for (const name of await readdir(dir).catch(() => []))
    if (/^(license|licence|copying|notice)(\.|$)/i.test(name)) {
      const content = await readFile(path.join(dir, name), 'utf8').catch(() => '');
      notices.push(`\n${dir} ${info.version}\n${content}`);
    }
}
await writeFile(path.join(root, 'THIRD_PARTY_NOTICES.txt'), notices.join('\n'));
await copyFile('../../LICENSE', path.join(root, 'LICENSE.txt'));
await copyFile(
  'schemas/studio_document.schema.json',
  path.join(root, 'studio_document.schema.json'),
);
await writeFile(
  path.join(root, 'dependency-inventory.json'),
  JSON.stringify({ format: 'mathkernel-studio-dependencies/1', packages }, null, 2),
);
async function files(dir) {
  const out = [];
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const file = path.join(dir, item.name);
    if (item.isDirectory()) out.push(...(await files(file)));
    else if (item.name !== 'asset-manifest.json') out.push(file);
  }
  return out;
}
const manifest = [];
for (const file of await files(root)) {
  const data = await readFile(file);
  manifest.push({
    path: path.relative(root, file).split(path.sep).join('/'),
    bytes: data.length,
    sha256: createHash('sha256').update(data).digest('hex'),
  });
}
await writeFile(
  path.join(root, 'asset-manifest.json'),
  JSON.stringify({ schema: 'studio-assets/1', files: manifest }, null, 2),
);
console.log(`Packaged ${manifest.length} local files and ${packages.length} dependency records.`);
