/** Development-only browser test double. Serves built assets and synthetic DTOs.
 * No credentials, kernel, filesystem API or real service can be configured here.
 * This configuration is outside the wheel; production uses StudioServer.
 */
import { defineConfig } from 'vite';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import fixtures from './qa/fixtures.json';
const root = path.resolve('host/src/mathkernel_studio/assets');
export default defineConfig({
  publicDir: false,
  server: { host: '0.0.0.0', allowedHosts: ['terminal.local'] },
  plugins: [
    {
      name: 'packaged-ui-qa',
      configureServer(server) {
        server.middlewares.use(async (req, res) => {
          const url = new URL(req.url ?? '/', 'http://qa.invalid');
          const route = url.pathname;
          const viewerPolicy =
            route === '/studio/viewer.html'
              ? await readFile(path.join(root, 'viewer-csp.txt'), 'utf8')
              : fixtures.csp;
          res.setHeader('Content-Security-Policy', viewerPolicy);
          res.setHeader('Cache-Control', 'no-store');
          res.setHeader('X-Content-Type-Options', 'nosniff');
          try {
            if (route.startsWith('/studio/api/')) {
              if (route === '/studio/api/session') {
                if (req.method !== 'POST') throw new Error('Invalid fixture session request');
                const body = await new Promise<string>((resolve, reject) => {
                  let value = '';
                  req.setEncoding('utf8');
                  req.on('data', (chunk) => {
                    value += chunk;
                    if (value.length > 1024) reject(new Error('Fixture request too large'));
                  });
                  req.on('end', () => resolve(value));
                  req.on('error', reject);
                });
                if (JSON.parse(body).code !== 'fixture-only') {
                  res.statusCode = 403;
                  res.end();
                  return;
                }
                res.setHeader(
                  'Set-Cookie',
                  'mkstudio_qa=1; HttpOnly; SameSite=Strict; Path=/studio',
                );
                res.statusCode = 204;
                res.end();
                return;
              }
              if (route === '/studio/api/session/disconnect') {
                if (req.method !== 'POST') throw new Error('Invalid fixture disconnect request');
                res.setHeader(
                  'Set-Cookie',
                  'mkstudio_qa=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/studio',
                );
                res.statusCode = 204;
                res.end();
                return;
              }
              const routes: Record<string, unknown> = {
                '/studio/api/handshake': fixtures.handshake,
                '/studio/api/catalog': fixtures.catalog,
                '/studio/api/results': fixtures.results,
                '/studio/api/results/fixture-result': fixtures.result,
                '/studio/api/objects': {
                  objects: [
                    {
                      object_id: 'fixture-matrix',
                      revision: 'fixture-1',
                      type_ref: 'Matrix',
                      summary: 'Synthetic 2×2 matrix',
                      shape: [2, 2],
                    },
                  ],
                  next_offset: null,
                },
                '/studio/api/runs': { runs: [], next_offset: null },
              };
              if (req.method !== 'GET' || !Object.hasOwn(routes, route)) {
                res.statusCode = 404;
                res.end('{"error":"Synthetic fixture route unavailable"}');
                return;
              }
              res.setHeader('Content-Type', 'application/json');
              res.end(
                JSON.stringify({
                  protocol: 'studio-host/1',
                  host_instance_id: 'fixture-host',
                  workspace_id: 'fixture-workspace',
                  request_id: req.headers['x-studio-request'],
                  observed_at: new Date().toISOString(),
                  payload: routes[route],
                }),
              );
              return;
            }
            const relative =
              ['/', '/studio/', '/studio'].includes(route) || route.includes('/workspaces/')
                ? 'index.html'
                : route.replace(/^\/studio\//, '');
            const file = path.resolve(root, relative);
            if (!file.startsWith(root + path.sep)) throw new Error('Invalid path');
            const media: Record<string, string> = {
              '.html': 'text/html',
              '.js': 'text/javascript',
              '.css': 'text/css',
              '.json': 'application/json',
            };
            if (!media[path.extname(file)]) throw new Error('Unsupported file');
            res.setHeader('Content-Type', media[path.extname(file)]!);
            res.end(await readFile(file));
          } catch {
            res.statusCode = 404;
            res.end('Fixture unavailable');
          }
        });
      },
    },
  ],
});
