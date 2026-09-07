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
import time
from urllib.parse import parse_qs, unquote, urlsplit
from .source import PROTOCOL, check_id

CSP = "; ".join(("default-src 'none'", "script-src 'self'", "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:", "font-src 'self'", "connect-src 'self'", "worker-src 'self'",
    "object-src 'none'", "base-uri 'none'", "frame-ancestors 'none'", "form-action 'self'"))
ASSETS = Path(__file__).parent / "assets"


class StudioServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, source, port=8765, assets=ASSETS):
        self.source = source
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
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
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
        if not self.origin_ok():
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
            try:
                query = parse_qs(parts.query, strict_parsing=True)
                if set(query) - {"offset"} or any(len(v) != 1 for v in query.values()):
                    raise ValueError()
                raw_offset = query.get("offset", ["0"])[0]
                if not re.fullmatch(r"[0-9]{1,9}", raw_offset):
                    raise ValueError()
                offset = int(raw_offset)
                source = self.server.source
                if path == "/studio/api/handshake": payload = source.handshake()
                elif path == "/studio/api/catalog": payload = source.catalog(offset)
                elif path == "/studio/api/results": payload = source.results(offset)
                elif match := re.fullmatch(r"/studio/api/results/([A-Za-z0-9_.:-]{1,128})(/page)?", path):
                    ref = check_id(match[1])
                    payload = source.result_page(ref, offset) if match[2] else source.result(ref)
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
    parser = argparse.ArgumentParser(description="MathKernel Studio — local catalog and authoring preview")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host-id", default=None, help="Stable operator-chosen identity, or a fresh identity per launch")
    parser.add_argument("--workspace-id", default="local")
    parser.add_argument("--test-host", action="store_true", help="Synthetic contract host; does not compute mathematics")
    from .testing import FAULTS, FixtureSource
    parser.add_argument("--fault", choices=FAULTS, default="none")
    args = parser.parse_args()
    if not ASSETS.joinpath("index.html").is_file():
        parser.error("Studio assets are missing. Build ui/studio with npm ci and npm run build before packaging.")
    if not 0 <= args.port <= 65535:
        parser.error("Port must be in 0–65535")
    if args.fault != "none" and not args.test_host:
        parser.error("--fault requires --test-host")
    if args.test_host:
        source = FixtureSource(args.fault)
    else:
        from mathkernel import MathKernel
        from .source import KernelSource
        source = KernelSource(MathKernel(), host_id=args.host_id or "local-" + secrets.token_hex(12), workspace_id=args.workspace_id)
    with StudioServer(source, args.port) as server:
        print(f"{'TEST HOST — synthetic fixtures. ' if source.test_host else ''}Open {server.origin}/studio/")
        print(f"One-time connection code (valid 10 minutes): {server.connect_code}")
        print("Loopback-only, read-only host; workflow execution unavailable. Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
