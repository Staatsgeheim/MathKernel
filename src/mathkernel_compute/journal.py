"""Independent compute journal; immutable payload hashes and transactional intent."""
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import sqlite3
import threading
from .protocol import canonical, decode

TABLES = frozenset({'jobs', 'attempts', 'plans', 'execution_specs', 'events', 'artifact_manifests',
    'artifact_refs', 'verification_reports', 'resource_leases', 'budget_grants',
    'budget_reservations', 'usage_entries', 'outbox', 'identity'})


class ComputeJournal:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.owner = open(self.root / 'coordinator.lock', 'a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.owner.seek(0); self.owner.write(b'0'); self.owner.flush(); self.owner.seek(0)
                msvcrt.locking(self.owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.owner.close()
            raise RuntimeError('Coordinator unavailable or state directory already owned')
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / 'compute.sqlite3', check_same_thread=False, isolation_level=None)
        try:
            self.db.execute('PRAGMA foreign_keys=ON')
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            with self.transaction():
                version = self.db.execute('PRAGMA user_version').fetchone()[0]
                if version not in (0, 1):
                    raise ValueError('JOURNAL_SCHEMA_UNSUPPORTED')
                for table in sorted(TABLES):
                    self.db.execute(f'CREATE TABLE IF NOT EXISTS {table} '
                                    '(ref TEXT PRIMARY KEY, value BLOB NOT NULL, sha256 TEXT NOT NULL)')
                self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS job_request ON jobs "
                                "(json_extract(value,'$.workspace_id'), json_extract(value,'$.client_request_id'))")
                self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS attempt_number ON attempts "
                                "(json_extract(value,'$.job_id'), json_extract(value,'$.attempt_number'))")
                self.db.execute('PRAGMA user_version=1')
        except BaseException:
            self.db.close(); self.owner.close()
            raise

    @contextmanager
    def transaction(self):
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                yield
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    def put(self, table, ref, value):
        if table not in TABLES:
            raise ValueError('Unknown journal table')
        raw = canonical(value)
        self.db.execute(f'INSERT INTO {table} VALUES(?,?,?) ON CONFLICT(ref) DO UPDATE '
                        'SET value=excluded.value, sha256=excluded.sha256',
                        (ref, raw, hashlib.sha256(raw).hexdigest()))

    def get(self, table, ref):
        if table not in TABLES:
            raise ValueError('Unknown journal table')
        with self.lock:
            row = self.db.execute(f'SELECT value,sha256 FROM {table} WHERE ref=?', (ref,)).fetchone()
        if row is None:
            raise KeyError(ref)
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise ValueError('JOURNAL_INTEGRITY_ERROR')
        return decode(row[0])

    def all(self, table):
        if table not in TABLES:
            raise ValueError('Unknown journal table')
        with self.lock:
            refs = self.db.execute(f'SELECT ref FROM {table} ORDER BY rowid').fetchall()
            return [self.get(table, row[0]) for row in refs]

    def close(self):
        self.db.close()
        self.owner.close()
