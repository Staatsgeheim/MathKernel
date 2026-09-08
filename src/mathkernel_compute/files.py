"""Private spool writes and content-addressed quarantine; no worker-selected paths."""
import hashlib
import os
from pathlib import Path
import secrets
import stat
from .protocol import MAX_MESSAGE


def atomic_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name('.pending-' + secrets.token_hex(16))
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def bounded_read(path, limit=MAX_MESSAGE):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('ARTIFACT_LIMIT_OR_TYPE')
        data = f.read(limit + 1)
        if len(data) > limit:
            raise ValueError('ARTIFACT_LIMIT')
        return data


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def put(self, raw, limit=MAX_MESSAGE):
        if len(raw) > limit:
            raise ValueError('ARTIFACT_LIMIT')
        identity = hashlib.sha256(raw).hexdigest()
        atomic_write(self.root / identity, raw)
        return identity

    def get(self, identity):
        if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
            raise ValueError('ARTIFACT_REFERENCE_INVALID')
        raw = bounded_read(self.root / identity)
        if hashlib.sha256(raw).hexdigest() != identity:
            raise ValueError('ARTIFACT_INTEGRITY_ERROR')
        return raw
