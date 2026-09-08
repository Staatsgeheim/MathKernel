"""Bounded data-only frames and RFC 8785 content identities."""
from __future__ import annotations
import hashlib
import json
import struct
from typing import TypeVar
from pydantic import BaseModel

MAX_MESSAGE = 1_048_576
T = TypeVar('T', bound=BaseModel)


def _check(value, depth=0):
    if depth > 64:
        raise ValueError('PROTOCOL_LIMIT: nesting exceeds 64')
    if isinstance(value, str):
        value.encode('utf-8', errors='strict')
    elif value is None or type(value) is bool:
        pass
    elif type(value) is int:
        if abs(value) > 2**53 - 1:
            raise ValueError('PROTOCOL_LIMIT: mathematical integers must be strings')
    elif type(value) is float:
        raise ValueError('PROTOCOL_SCHEMA: decimal quantities must be strings')
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check(item, depth + 1)
    elif type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str) or key in {'__proto__', 'constructor', 'prototype'}:
                raise ValueError('PROTOCOL_SCHEMA: invalid object key')
            _check(key, depth + 1)
            _check(item, depth + 1)
    else:
        raise ValueError('PROTOCOL_SCHEMA: unsupported data type')


def canonical(value) -> bytes:
    # Optional compute dependency. Base MathKernel import never loads it.
    import rfc8785
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    _check(value)
    return rfc8785.dumps(value)


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def decode(raw: bytes, limit=MAX_MESSAGE):
    if not isinstance(raw, bytes) or len(raw) > limit:
        raise ValueError('PROTOCOL_LIMIT: message byte budget exceeded')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('PROTOCOL_SCHEMA: duplicate key')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('PROTOCOL_SCHEMA: nonfinite value')
    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=bad_constant)
        _check(value)
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError('PROTOCOL_SCHEMA: invalid JSON') from exc


def parse(model: type[T], raw: bytes, limit=MAX_MESSAGE) -> T:
    return model.model_validate_json(canonical(decode(raw, limit)))


def read_frame(stream, limit=MAX_MESSAGE) -> bytes:
    def exact(n):
        parts = bytearray()
        while len(parts) < n:
            part = stream.read(n - len(parts))
            if not part:
                raise ValueError('OUTPUT_INCOMPLETE: truncated frame')
            parts.extend(part)
        return bytes(parts)
    size = struct.unpack('!I', exact(4))[0]
    if size > limit:
        raise ValueError('PROTOCOL_LIMIT: declared frame too large')
    raw = exact(size)
    decode(raw, limit)
    return raw


def write_frame(stream, value, limit=MAX_MESSAGE):
    raw = canonical(value)
    if len(raw) > limit:
        raise ValueError('PROTOCOL_LIMIT: frame too large')
    stream.write(struct.pack('!I', len(raw)) + raw)
    stream.flush()
