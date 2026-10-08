"""One CLI for inspect, fingerprint, diagnostic probe, and authorized replay."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from . import (AuditLimits, ComparatorRequest, FormalProjectSpec, audit_lean_project,
               inventory, lean_probe, verify_with_comparator)
from .correspondence import (
    LeanExtractionRequest, audit_correspondence, extract_lean_signature,
)
from .source import bounded_bytes


def _json_file(path):
    return json.loads(bounded_bytes(Path(path), 1024**2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MathKernel formal-project audit. Inspection never executes project code.")
    sub = parser.add_subparsers(dest="action", required=True)
    inspect = sub.add_parser("inspect", help="Bounded source/config inspection; no proof validation")
    inspect.add_argument("root"); inspect.add_argument("--spec"); inspect.add_argument("--output")
    finger = sub.add_parser("fingerprint", help="Fingerprint a reviewed reference workspace; does not make it trustworthy")
    finger.add_argument("root"); finger.add_argument("--include-dependencies", action="store_true")
    probe = sub.add_parser("probe", help="Emit an unexecuted Lean diagnostic, not a proof certificate")
    probe.add_argument("spec"); probe.add_argument("--output")
    replay = sub.add_parser("replay", help="Run pinned Comparator/nanoda in a clean sandboxed workspace")
    replay.add_argument("request"); replay.add_argument("--authorize-execution", action="store_true")
    replay.add_argument("--output")
    correspond = sub.add_parser(
        "correspond",
        help="Compare a reviewed source contract with a pinned Lean declaration",
    )
    correspond.add_argument("--project", required=True)
    correspond.add_argument("--formal-spec", required=True)
    correspond.add_argument("--manifest", required=True)
    correspond.add_argument("--document")
    correspond.add_argument("--excerpt")
    correspond.add_argument("--extractor-request")
    correspond.add_argument("--inspection-only", action="store_true")
    correspond.add_argument("--authorize-execution", action="store_true")
    correspond.add_argument("--backend", choices=("python-reference",), default="python-reference")
    correspond.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            spec = FormalProjectSpec.model_validate(_json_file(args.spec)) if args.spec else None
            result = audit_lean_project(args.root, spec)
            content = result.model_dump_json(indent=2) + "\n"
            code = 2 if any(f.severity == "error" for f in result.findings) else 0
        elif args.action == "fingerprint":
            limits = AuditLimits(max_files=200000, max_file_bytes=1024**3, max_total_bytes=32 * 1024**3) if args.include_dependencies else None
            root, files, digest = inventory(args.root, limits, include_dependencies=args.include_dependencies)
            content = json.dumps({"sha256": digest, "files": len(files), "bytes": sum(f.size for f in files),
                                  "includes_dependencies": args.include_dependencies, "trust": "unknown"}, indent=2) + "\n"
            code = 0
        elif args.action == "probe":
            content = lean_probe(FormalProjectSpec.model_validate(_json_file(args.spec)))
            code = 0
        elif args.action == "replay":
            result = verify_with_comparator(ComparatorRequest.model_validate(_json_file(args.request)),
                                            authorize_execution=args.authorize_execution)
            content = result.model_dump_json(indent=2) + "\n"
            code = 0 if result.status == "accepted" else 2
        else:
            formal_spec = FormalProjectSpec.model_validate(_json_file(args.formal_spec))
            extractor = None
            if not args.inspection_only:
                if not args.extractor_request:
                    raise ValueError("--extractor-request is required outside inspection-only mode")
                request = LeanExtractionRequest.model_validate(_json_file(args.extractor_request))
                extractor = lambda **_: extract_lean_signature(
                    request, authorize_execution=args.authorize_execution
                )
            result = audit_correspondence(
                project_root=args.project,
                formal_spec=formal_spec,
                correspondence_spec=args.manifest,
                extractor=extractor,
                document_path=args.document,
                excerpt_path=args.excerpt,
                inspection_only=args.inspection_only,
                backend=args.backend,
            )
            content = result.model_dump_json(indent=2) + "\n"
            code = 0 if result.status in {"matched", "incomplete"} else 2
        output = getattr(args, "output", None)
        if output:
            # Do not overwrite an input, trusted reference or existing report.
            with Path(output).open("x", encoding="utf-8") as stream:
                stream.write(content)
        else:
            sys.stdout.write(content)
        return code
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "error", "trust": "unknown", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
