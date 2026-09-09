"""Bounded read-only inspection of Lean sources. No process or network calls."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from .models import (AuditFinding, AuditLimits, FormalAuditReport, FormalProjectSpec,
                     SourceFile, TOOLCHAIN, lean_name)

SOURCE_EXCLUDED = frozenset({".git", ".lake", ".venv", "venv", "node_modules", "__pycache__"})


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or any(ord(c) < 32 for c in value):
        raise ValueError("Expected a printable relative path")
    path = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or path.is_absolute()
            or any(p in {".", "..", ""} for p in value.split("/"))):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def checked_root(root: str | Path) -> Path:
    path = Path(root).absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink roots/parents are not allowed: {parent}")
    if not path.is_dir():
        raise ValueError("Project root is not an existing directory")
    return path.resolve()


def bounded_bytes(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"Not a bounded regular file: {path}")
        value = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
        if len(value) > maximum or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"File changed or exceeded the limit while reading: {path}")
        return value


def hash_file(path: Path, maximum: int) -> tuple[int, str]:
    """Streaming file fingerprint with regular-file and mutation checks."""
    if maximum < 0:
        raise ValueError("Project exceeds max_total_bytes")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ValueError(f"Not a bounded regular file: {path}")
        digest, count = hashlib.sha256(), 0
        while block := stream.read(min(1024**2, maximum - count + 1)):
            count += len(block)
            if count > maximum:
                raise ValueError(f"File exceeded size limit: {path}")
            digest.update(block)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError(f"File changed during fingerprinting: {path}")
        return count, digest.hexdigest()


def inventory(root: str | Path, limits: AuditLimits | None = None, *,
              include_dependencies: bool = False) -> tuple[Path, tuple[SourceFile, ...], str]:
    root = checked_root(root)
    limits = limits or AuditLimits()
    items: list[SourceFile] = []
    total = 0
    excluded = {".git"} if include_dependencies else SOURCE_EXCLUDED
    for current, dirs, files in os.walk(root, followlinks=False):
        current = Path(current)
        for name in (*dirs, *files):
            if (current / name).is_symlink():
                raise ValueError(f"Symlink in project: {(current / name).relative_to(root)}")
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(files):
            if name in excluded:
                continue
            path = current / name
            relative = path.relative_to(root).as_posix()
            safe_relative(relative)
            if len(items) >= limits.max_files:
                raise ValueError("Project exceeds max_files; audit aborted, not partially accepted")
            size, digest = hash_file(path, min(limits.max_file_bytes, limits.max_total_bytes - total))
            total += size
            if total > limits.max_total_bytes:
                raise ValueError("Project exceeds max_total_bytes")
            items.append(SourceFile(path=relative, size=size, sha256=digest))
    items.sort(key=lambda f: f.path)
    encoded = json.dumps([f.model_dump() for f in items], ensure_ascii=True,
                         sort_keys=True, separators=(",", ":")).encode()
    return root, tuple(items), hashlib.sha256(encoded).hexdigest()


def mask_lean(text: str) -> str:
    """Erase comments/literals but preserve offsets and newlines.

    Nested block comments, escaped strings, raw strings and quoted identifiers
    are handled. Quoted identifiers are masked rather than interpreted; this
    scanner deliberately does not claim to understand Lean's extensible grammar.
    """
    out = list(text)
    i, n = 0, len(text)
    def erase(a: int, b: int):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "
    while i < n:
        start = i
        if text.startswith("--", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            erase(start, i)
        elif text.startswith("/-", i):
            depth = 1
            i += 2
            while i < n and depth:
                if text.startswith("/-", i): depth, i = depth + 1, i + 2
                elif text.startswith("-/", i): depth, i = depth - 1, i + 2
                else: i += 1
            if depth:
                raise ValueError("Unterminated Lean block comment")
            erase(start, i)
        elif text[i] == '"' or (text[i] == "r" and (m := re.match(r'r(#+)?"', text[i:]))):
            raw = text[i] == "r"
            hashes = (m.group(1) or "") if raw else ""
            i += len(m.group(0)) if raw else 1
            ending = '"' + hashes
            while i < n:
                if text.startswith(ending, i):
                    i += len(ending)
                    break
                if not raw and text[i] == "\\": i += 2
                else: i += 1
            else:
                raise ValueError("Unterminated Lean string")
            erase(start, i)
        elif text[i] == "«":
            j = text.find("»", i + 1)
            if j < 0:
                raise ValueError("Unterminated Lean quoted identifier")
            i = j + 1
            erase(start, i)
        else:
            i += 1
    return "".join(out)


def imports(masked: str) -> tuple[str, ...]:
    result = []
    for m in re.finditer(r"(?m)^\s*(?:public\s+|private\s+|meta\s+)*import\s+([^\n]+)", masked):
        for word in m.group(1).split():
            try: result.append(lean_name(word))
            except ValueError: raise ValueError(f"Unsupported import syntax: {word!r}") from None
    return tuple(dict.fromkeys(result))


def declarations(masked: str) -> dict[str, str]:
    """Index simple top-level declaration headers for diagnostics, not elaboration."""
    stack: list[tuple[str, str]] = []
    found: dict[str, str] = {}
    name_pattern = r"[A-Za-z_][A-Za-z0-9_'.]*"
    for line in masked.splitlines():
        section = re.match(r"\s*(namespace|section)\b(?:\s+(" + name_pattern + r"))?", line)
        if section:
            stack.append((section[1], section[2] or ""))
            continue
        if re.match(r"\s*end\b", line):
            if stack: stack.pop()
            continue
        declaration = re.match(r"\s*(?:(?:noncomputable|private|protected|unsafe|partial)\s+)*"
                               r"(theorem|lemma|def|abbrev|opaque|axiom|structure|class)\s+(" + name_pattern + r")", line)
        if declaration:
            prefix = ".".join(name for kind, name in stack if kind == "namespace")
            found[(prefix + "." if prefix else "") + declaration[2]] = declaration[1]
    return found


def validate_comparator_config(config: dict, spec: FormalProjectSpec) -> list[str]:
    errors = []
    if not isinstance(config, dict):
        return ["Comparator config must be a JSON object"]
    # Arbitrary external-kernel commands in a submission are never executed.
    allowed = {"challenge_module", "solution_module", "theorem_names", "permitted_axioms", "enable_nanoda"}
    unexpected = set(config) - allowed
    if unexpected:
        errors.append("Unsupported/unsafe Comparator keys: " + ", ".join(sorted(unexpected)))
    if config.get("enable_nanoda") is not True:
        errors.append("Independent nanoda checking must explicitly be enabled")
    for key in ("challenge_module", "solution_module"):
        try: lean_name(config.get(key, ""))
        except (ValueError, TypeError): errors.append(f"Invalid {key}")
    if config.get("challenge_module") == config.get("solution_module"):
        errors.append("The challenge and solution modules must differ")
    names = config.get("theorem_names")
    expected = [t.declaration for t in spec.targets]
    if (not isinstance(names, list) or not names or not all(isinstance(v, str) for v in names)
            or len(names) != len(set(names))):
        errors.append("Nonempty, duplicate-free theorem_names are required")
    elif set(names) != set(expected):
        errors.append("theorem_names do not match the operator's requested targets")
    ax = config.get("permitted_axioms")
    if (not isinstance(ax, list) or not all(isinstance(v, str) for v in ax)
            or len(ax) != len(set(ax)) or set(ax) != set(spec.permitted_axioms)):
        errors.append("permitted_axioms do not match the operator's axiom policy")
    if any(t.module != config.get("solution_module") for t in spec.targets):
        errors.append("Requested target modules do not match solution_module")
    return errors


def audit_lean_project(root: str | Path, spec: FormalProjectSpec | None = None,
                       limits: AuditLimits | None = None) -> FormalAuditReport:
    spec, limits = spec or FormalProjectSpec(), limits or AuditLimits()
    root, files, fingerprint = inventory(root, limits)
    findings: list[AuditFinding] = []
    capped = False
    def note(code, severity, path, message, line=0, scope="source_only"):
        nonlocal capped
        if len(findings) >= limits.max_findings:
            capped = True
            return
        findings.append(AuditFinding(code=code, severity=severity, path=path,
                                     message=message, line=line, scope=scope))
    by_path = {f.path: f for f in files}
    def text(path):
        if path not in by_path:
            raise ValueError(f"Missing project file: {path}")
        data = bounded_bytes(root / path, limits.max_file_bytes)
        if hashlib.sha256(data).hexdigest() != by_path[path].sha256:
            raise ValueError("Source changed during inspection")
        return data.decode("utf-8")
    if spec.expected_source_sha256 and fingerprint != spec.expected_source_sha256:
        note("SOURCE_PIN_MISMATCH", "error", "", "Source fingerprint differs from the requested pin")
    toolchain = text("lean-toolchain").strip() if "lean-toolchain" in by_path else None
    if not toolchain or not TOOLCHAIN.fullmatch(toolchain):
        note("TOOLCHAIN_UNPINNED", "error", "lean-toolchain", "Missing or unsupported concrete toolchain pin")
    if spec.expected_toolchain and toolchain != spec.expected_toolchain:
        note("TOOLCHAIN_MISMATCH", "error", "lean-toolchain", "Toolchain differs from the requested version")
    pins = []
    if "lake-manifest.json" not in by_path:
        note("DEPENDENCIES_UNLOCKED", "warning", "lake-manifest.json", "Dependency lockfile is absent")
    else:
        try:
            lock = json.loads(text("lake-manifest.json"))
            packages = lock["packages"]
            if not isinstance(packages, list): raise ValueError("packages is not a list")
            for item in packages:
                if not isinstance(item, dict): raise ValueError("dependency is not an object")
                pin = {k: item.get(k) for k in ("name", "type", "url", "rev", "inputRev")}
                pins.append(pin)
                if item.get("type") != "git" or not re.fullmatch(r"[0-9a-f]{40}", str(item.get("rev", ""))):
                    note("DEPENDENCY_UNPINNED", "error", "lake-manifest.json", f"Unpinned dependency: {item.get('name')}")
        except (ValueError, KeyError, TypeError) as exc:
            note("LOCKFILE_INVALID", "error", "lake-manifest.json", str(exc))
    graph, indexed = {}, {}
    for file in files:
        if not file.path.endswith(".lean"): continue
        module = file.path[:-5].replace("/", ".")
        try:
            masked = mask_lean(text(file.path))
            graph[module] = imports(masked)
            indexed[module] = declarations(masked)
            if re.search(r'\b[si]!\s*"', text(file.path)):
                note("INTERPOLATED_SYNTAX", "warning", file.path,
                     "Code inside interpolated strings is not analyzed by the lexical scanner")
            patterns = (
                (r"\b(?:sorry|admit|sorryAx)\b", "PLACEHOLDER", "warning", "Proof placeholder; target dependency requires kernel-level checking"),
                (r"(?m)^\s*axiom\s+", "AXIOM_DECLARATION", "warning", "Source axiom; check the target's transitive axiom dependencies"),
                (r"\b(?:native_decide|ofReduceBool)\b", "NATIVE_PROOF", "warning", "Native-evaluation proof route requires an explicit trust audit"),
                (r"\b(?:unsafe|implemented_by|extern)\b", "NATIVE_CODE", "warning", "Native/unsafe code may execute during project elaboration or build"),
                (r"\b(?:debug\.skipKernelTC|debug\.skipKernelCheck)\b", "KERNEL_CHECK_OVERRIDE", "error", "Kernel-check override in source"),
                (r"(?m)^\s*(?:macro|syntax|elab|initialize|run_elab)\b", "ELABORATOR_EXTENSION", "warning", "Executable/extensible Lean syntax; lexical inspection is incomplete"),
            )
            for pattern, code, severity, message in patterns:
                for match in re.finditer(pattern, masked):
                    note(code, severity, file.path, message, masked.count("\n", 0, match.start()) + 1)
        except (ValueError, UnicodeError) as exc:
            note("SOURCE_UNPARSED", "error", file.path, str(exc))
    closure, unresolved, pending = set(), set(), [t.module for t in spec.targets]
    while pending:
        module = pending.pop()
        if module in closure: continue
        if module not in graph:
            unresolved.add(module)
            continue
        closure.add(module)
        pending.extend(graph[module])
    for target in spec.targets:
        if target.module not in graph:
            note("TARGET_MODULE_MISSING", "error", target.module.replace(".", "/") + ".lean", "Requested module is not in the inspected source")
        elif target.declaration not in indexed.get(target.module, {}):
            note("TARGET_NOT_INDEXED", "warning", target.module.replace(".", "/") + ".lean", "Declaration not found by the limited source index; Lean #check is still required")
        elif indexed[target.module][target.declaration] not in {"theorem", "lemma"}:
            note("TARGET_NOT_THEOREM", "warning", target.module.replace(".", "/") + ".lean", "Requested name is not indexed as a theorem or lemma")
    for config_path in spec.comparator_configs:
        safe_relative(config_path)
        try:
            config = json.loads(text(config_path))
            for error in validate_comparator_config(config, spec):
                note("COMPARATOR_POLICY", "error", config_path, error)
            if isinstance(config, dict):
                challenge = config.get("challenge_module", "")
                if challenge in closure:
                    note("CHALLENGE_IMPORTED", "warning", config_path, "Solution import closure includes the challenge; inspect actual theorem dependencies")
        except (ValueError, TypeError, UnicodeError) as exc:
            note("COMPARATOR_CONFIG_INVALID", "error", config_path, str(exc))
    # A second inventory also binds files which did not need text parsing.
    if inventory(root, limits)[2] != fingerprint:
        note("SOURCE_CHANGED", "error", "", "Source changed while the audit was running")
    if capped:
        findings.append(AuditFinding(code="FINDINGS_TRUNCATED", severity="error", path="",
                                     message="Finding limit reached; report is incomplete"))
    status = "incomplete" if capped or any(f.code in {"SOURCE_CHANGED", "SOURCE_UNPARSED"} for f in findings) else "issues_found" if findings else "inspected"
    return FormalAuditReport(status=status, source_sha256=fingerprint, source_files=files,
        toolchain=toolchain, local_imports=graph, target_import_closure=tuple(sorted(closure)),
        unresolved_imports=tuple(sorted(unresolved)), findings=tuple(findings), dependency_pins=tuple(pins))


def lean_probe(spec: FormalProjectSpec) -> str:
    """Emit an UNEXECUTED diagnostic candidate, never a certificate.

    This must not be run against untrusted submissions before Comparator.
    Human-readable #print output is not accepted as cryptographic proof evidence.
    """
    if not spec.targets:
        raise ValueError("At least one target is required")
    lines = ["-- DIAGNOSTIC ONLY. Do not run on untrusted code before Comparator."]
    lines += ["import " + m for m in sorted({t.module for t in spec.targets})]
    for target in spec.targets:
        lines += ["#check " + target.declaration, "#print axioms " + target.declaration]
    return "\n".join(lines) + "\n"
