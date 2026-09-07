"""Loopback-only optional asset/inspection facade. No operation-invocation route."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import secrets
import hashlib
import time
from urllib.parse import parse_qs, unquote, urlsplit
from .source import PROTOCOL, check_id
from .services import ACTION_FIELDS, ACTION_FEATURES, SessionScope, parse_command

CSP = "; ".join(("default-src 'none'", "script-src 'self'", "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:", "font-src 'self'", "connect-src 'self'", "worker-src 'self'",
    "object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'", "frame-src 'self'", "form-action 'self'"))
ASSETS = Path(__file__).parent / "assets"
VIEWER_CSP = (ASSETS / "viewer-csp.txt").read_text(encoding="ascii")


class StudioServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, source, port=8765, assets=ASSETS, service=None):
        self.source = source
        self.service = service
        self.rate_window = time.monotonic()
        self.rate_count = 0
        self.assets = Path(assets).resolve()
        self.connect_code = secrets.token_urlsafe(24)
        self.code_expires = time.monotonic() + 600
        self.session = None
        self.session_expires = 0.0
        self.failed_connects = 0
        super().__init__(("127.0.0.1", port), Handler)
        self.authority = f"127.0.0.1:{self.server_port}"
        self.origin = f"http://{self.authority}"

    def get_request(self):
        sock, address = super().get_request()
        sock.settimeout(3)
        return sock, address


class Handler(BaseHTTPRequestHandler):
    server: StudioServer
    server_version = "MathKernelStudio"

    def log_message(self, format, *args):
        # No expressions, references, credentials or response bodies in access logs.
        pass

    def respond(self, status, payload=b"", media="application/json", cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(payload)))
        isolated = self.path == '/studio/viewer.html'
        self.send_header("Content-Security-Policy", VIEWER_CSP if isolated else CSP)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "SAMEORIGIN" if isolated else "DENY")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(payload)

    def error(self, status, code):
        self.respond(status, json.dumps({"error": code}).encode())

    def origin_ok(self, mutation=False):
        if self.headers.get_all("Host") != [self.server.authority]:
            return False
        origins = self.headers.get_all("Origin") or []
        if origins and origins != [self.server.origin]:
            return False
        if mutation and origins != [self.server.origin]:
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site")
        return fetch_site not in {"cross-site", "same-site"}

    def authenticated(self):
        if not self.server.session or time.monotonic() >= self.server.session_expires:
            return False
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
            value = cookie.get("mkstudio_session")
            return value is not None and secrets.compare_digest(value.value, self.server.session)
        except Exception:
            return False

    def service_scope(self):
        return SessionScope(self.server.source.host_id, self.server.source.workspace_id,
            hashlib.sha256(self.server.session.encode()).hexdigest())

    def rate_ok(self):
        now = time.monotonic()
        if now - self.server.rate_window >= 10:
            self.server.rate_window, self.server.rate_count = now, 0
        self.server.rate_count += 1
        if self.server.rate_count > 120:
            self.error(429, 'request_rate_limit')
            return False
        return True

    def envelope(self, payload):
        request_id = self.headers.get("X-Studio-Request", "")
        try:
            check_id(request_id)
        except ValueError:
            self.error(400, "invalid_request_id")
            return
        scope = self.server.source
        envelope = dict(protocol=PROTOCOL, host_instance_id=scope.host_id,
            workspace_id="wrong-workspace" if getattr(scope, "fault", "") == "scope_mismatch" else scope.workspace_id,
            request_id=request_id, observed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), payload=payload)
        data = json.dumps(envelope, ensure_ascii=True, allow_nan=False).encode("ascii")
        if len(data) > 2097152:
            self.error(413, "control_response_limit")
        else:
            self.respond(200, data)

    def do_POST(self):
        if not self.origin_ok(mutation=True):
            self.error(403, "origin_rejected")
            return
        if not self.rate_ok(): return
        if self.path == '/studio/api/session/disconnect':
            if not self.authenticated() or self.headers.get('X-Studio-Action') != 'disconnect':
                self.error(403, 'session_required'); return
            self.server.session = None
            self.respond(200, b'{"disconnected":true}', cookie='mkstudio_session=; HttpOnly; SameSite=Strict; Path=/studio/; Max-Age=0')
            return
        action = self.path.removeprefix('/studio/api/')
        if self.server.service is not None and action in ACTION_FIELDS:
            if not self.authenticated(): self.error(401, 'session_required'); return
            if not self.server.service.capabilities()['features'].get(ACTION_FEATURES[action], False):
                self.error(403, 'feature_unavailable'); return
            if self.headers.get('Content-Type') != 'application/json' or self.headers.get('X-Studio-Action') != action:
                self.error(415, 'invalid_command_request'); return
            lengths = self.headers.get_all('Content-Length') or []
            if len(lengths) != 1 or not re.fullmatch(r'[0-9]{1,8}', lengths[0]) or self.headers.get('Transfer-Encoding'):
                self.error(400, 'invalid_length'); return
            length = int(lengths[0])
            if not 0 < length <= 10 * 1024 * 1024 + 8192: self.error(413, 'command_budget'); return
            try:
                request_id = check_id(self.headers.get('X-Studio-Request', ''))
                payload = parse_command(self.rfile.read(length), action)
                if 'client_request_id' in payload and payload['client_request_id'] != request_id: raise ValueError('Correlation mismatch')
                self.envelope(self.server.service.command(action, payload, request_id, self.service_scope()))
            except PermissionError: self.error(403, 'service_permission_denied')
            except KeyError: self.error(404, 'unavailable_or_expired')
            except (ValueError, TypeError, RecursionError): self.error(400, 'invalid_command_or_contract')
            except Exception: self.error(500, 'command_outcome_unknown')
            return
        if self.path != "/studio/api/session":
            self.error(405, "read_only_host")
            return
        if self.headers.get("Content-Type") != "application/json" or self.headers.get("X-Studio-Action") != "connect":
            self.error(415, "invalid_session_request")
            return
        lengths = self.headers.get_all("Content-Length") or []
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,4}", lengths[0]) or self.headers.get("Transfer-Encoding"):
            self.error(400, "invalid_length")
            return
        length = int(lengths[0])
        if not 1 <= length <= 1024:
            self.error(413, "session_request_limit")
            return
        try:
            pairs = json.loads(self.rfile.read(length), object_pairs_hook=lambda p: p)
            if not isinstance(pairs, list) or len(pairs) != 1 or pairs[0][0] != "code" or not isinstance(pairs[0][1], str):
                raise ValueError()
            code = pairs[0][1]
        except (ValueError, TypeError, IndexError):
            self.error(400, "invalid_session_request")
            return
        if (not code.isascii() or not self.server.connect_code or time.monotonic() >= self.server.code_expires or self.server.failed_connects >= 10
                or not secrets.compare_digest(code, self.server.connect_code)):
            self.server.failed_connects += 1
            self.error(401, "code_rejected")
            return
        self.server.connect_code = None
        self.server.session = secrets.token_urlsafe(32)
        self.server.session_expires = time.monotonic() + 3600
        self.respond(200, b'{"connected":true}', cookie=f"mkstudio_session={self.server.session}; HttpOnly; SameSite=Strict; Path=/studio/; Max-Age=3600")

    def do_GET(self):
        # Only immutable public viewer assets are readable by the opaque frame.
        viewer_asset = self.path in {'/studio/viewer.css', '/studio/viewer/viewer.js'}
        opaque_asset = viewer_asset and self.headers.get_all('Host') == [self.server.authority] and (self.headers.get_all('Origin') or []) in ([], ['null'])
        if not self.origin_ok() and not opaque_asset:
            self.error(403, "origin_rejected")
            return
        parts = urlsplit(self.path)
        if parts.scheme or parts.netloc:
            self.error(400, "invalid_target")
            return
        path = unquote(parts.path)
        if path.startswith("/studio/api/"):
            if not self.authenticated():
                self.error(401, "session_required")
                return
            if not self.rate_ok(): return
            try:
                query = parse_qs(parts.query, strict_parsing=True)
                if set(query) - {"offset"} or any(len(v) != 1 for v in query.values()):
                    raise ValueError()
                raw_offset = query.get("offset", ["0"])[0]
                if not re.fullmatch(r"[0-9]{1,9}", raw_offset):
                    raise ValueError()
                offset = int(raw_offset)
                source = self.server.source
                if path == "/studio/api/handshake":
                    payload = source.handshake()
                    if self.server.service is not None:
                        extension = self.server.service.capabilities()
                        payload['features'] = {**payload['features'], **extension['features']}
                        payload['extensions'] = extension['extensions']
                elif path == "/studio/api/catalog": payload = source.catalog(offset)
                elif path == "/studio/api/results": payload = source.results(offset)
                elif match := re.fullmatch(r"/studio/api/results/([A-Za-z0-9_.:-]{1,128})(/page|/admission)?", path):
                    ref = check_id(match[1])
                    if match[2] == '/admission':
                        if not hasattr(source, 'result_admission'): raise KeyError('Admission snapshot unavailable')
                        payload = source.result_admission(ref)
                    else: payload = source.result_page(ref, offset) if match[2] else source.result(ref)
                elif self.server.service is not None and (match := re.fullmatch(r'/studio/api/(runs|requests|objects|subworkflows)(?:/([A-Za-z0-9_.:-]{1,128}))?', path)):
                    kind, ref = match[1], match[2]
                    if ref: check_id(ref)
                    payload = self.server.service.read(kind, ref, offset, self.service_scope())
                else:
                    self.error(404, "unsupported_route")
                    return
                self.envelope(payload)
            except PermissionError: self.error(403, "permission_denied")
            except KeyError: self.error(404, "unavailable_or_expired")
            except (ValueError, TypeError): self.error(400, "invalid_request_or_contract")
            except Exception: self.error(500, "host_read_failed")
            return
        if path in {"/studio", "/studio/"}:
            target = self.server.assets / "index.html"
        elif path in {'/studio/viewer.html', '/studio/viewer.css', '/studio/viewer/viewer.js'}:
            target = self.server.assets / path.removeprefix('/studio/')
        elif path.startswith("/studio/assets/"):
            target = (self.server.assets / path.removeprefix("/studio/")).resolve()
            if not target.is_relative_to(self.server.assets):
                self.error(404, "not_found")
                return
        elif re.fullmatch(r"/studio/workspaces/[A-Za-z0-9_.:-]+/(documents|runs|artifacts)/[A-Za-z0-9_.:-]+", path):
            target = self.server.assets / "index.html"
        else:
            self.error(404, "not_found")
            return
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8", ".json": "application/json"}
        if not target.is_file() or target.suffix not in types:
            self.error(404, "asset_missing_rebuild_studio")
            return
        self.respond(200, target.read_bytes(), types[target.suffix])


def main():
    parser = argparse.ArgumentParser(description="MathKernel Studio — local workflows and evidence inspection")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument('--read-only', action='store_true', help='Disable the local workflow owner')
    parser.add_argument('--state-dir', type=Path, default=Path.home()/'.mathkernel'/'studio', help='Durable local workflow state directory; one owner per directory')
    parser.add_argument('--run-timeout', type=int, default=60, help='Local process wall-time limit in seconds')
    parser.add_argument('--memory-mb', type=int, default=0, help='Optional POSIX process address-space limit; zero means no hard memory cap')
    parser.add_argument('--deny-execution', action='store_true', help='Expose planning with execution denied by operator policy')
    parser.add_argument("--host-id", default=None, help="Operator identity; otherwise persisted by the local workflow owner")
    parser.add_argument("--workspace-id", default="local")
    parser.add_argument("--test-host", action="store_true", help="Synthetic contract host; does not compute mathematics")
    parser.add_argument('--test-workflow', action='store_true', help='Enable synthetic workflow UI fixtures; requires --test-host')
    parser.add_argument('--workflow-fault', choices=['none', 'submission_unknown', 'stale_plan', 'policy_denied'], default='none')
    from .testing import FAULTS, FixtureSource
    parser.add_argument("--fault", choices=FAULTS, default="none")
    args = parser.parse_args()
    if not ASSETS.joinpath("index.html").is_file():
        parser.error("Studio assets are missing. Build ui/studio with npm ci and npm run build before packaging.")
    if not 0 <= args.port <= 65535:
        parser.error("Port must be in 0–65535")
    if args.fault != "none" and not args.test_host:
        parser.error("--fault requires --test-host")
    if args.test_workflow and not args.test_host:
        parser.error('--test-workflow requires --test-host')
    if args.workflow_fault != 'none' and not args.test_workflow:
        parser.error('--workflow-fault requires --test-workflow')
    runtime = None
    if args.test_host:
        source = FixtureSource(args.fault)
        from .workflow_testing import FixtureWorkflowService
        service = FixtureWorkflowService(args.workflow_fault) if args.test_workflow else None
    else:
        from mathkernel import MathKernel
        kernel = MathKernel()
        if args.read_only:
            from .source import KernelSource
            source = KernelSource(kernel, host_id=args.host_id or 'local-'+secrets.token_hex(12), workspace_id=args.workspace_id)
            service = None
        else:
            from mathkernel_workflow import WorkflowRuntime, LocalPolicy
            from .local import LocalWorkflowSource
            runtime = WorkflowRuntime(args.state_dir, host_id=args.host_id, workspace_id=args.workspace_id,
                policy=LocalPolicy(wall_seconds=args.run_timeout, memory_mb=args.memory_mb, enabled=not args.deny_execution))
            service = runtime
            source = LocalWorkflowSource(kernel, runtime)
    try:
        with StudioServer(source, args.port, service=service) as server:
            print(f"{'TEST HOST — synthetic fixtures. ' if source.test_host else ''}Open {server.origin}/studio/")
            print(f"One-time connection code (valid 10 minutes): {server.connect_code}")
            print('Local workflow owner enabled; execution requires a reviewed plan and host approval. Ctrl+C to stop.' if runtime else 'Read-only or synthetic host. Ctrl+C to stop.')
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    main()
