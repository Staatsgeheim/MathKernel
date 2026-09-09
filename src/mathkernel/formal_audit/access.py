"""Operator-owned read allowlist for the MCP surface (not an OS sandbox)."""
from pathlib import Path
from .source import checked_root


def authorized_root(root: str, allowed_roots: tuple[str, ...]) -> Path:
    if not allowed_roots:
        raise PermissionError("Formal project file access is disabled. Set MATHKERNEL_FORMAL_PROJECT_ROOTS before starting MCP.")
    candidate = checked_root(root)
    for allowed in allowed_roots:
        trusted = checked_root(allowed)
        if candidate.is_relative_to(trusted):
            return candidate
    raise PermissionError("Project is outside the operator's formal-audit read allowlist")
