# =============================================================================
# MathKernel - bigint
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import math

_BASE = 1_000_000_000
_CHUNK = 9


def decimal_to_int(text: str) -> int:
    """Convert arbitrary-length decimal text without Python's int(str) digit limit."""
    s = text.strip()
    if not s:
        raise ValueError("empty integer")
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    if not s or not s.isdigit():
        raise ValueError("invalid decimal integer")
    s = s.lstrip("0") or "0"
    value = 0
    first = len(s) % _CHUNK or _CHUNK
    value = int(s[:first])
    for i in range(first, len(s), _CHUNK):
        value = value * _BASE + int(s[i:i+_CHUNK])
    return sign * value


def int_to_decimal(value: int) -> str:
    """Render arbitrary-size int without Python's int->str digit limit."""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    n = -value if value < 0 else value
    chunks: list[int] = []
    while n:
        n, rem = divmod(n, _BASE)
        chunks.append(rem)
    head = str(chunks.pop())
    tail = "".join(f"{x:09d}" for x in reversed(chunks))
    return sign + head + tail


def decimal_digit_count(value: int) -> int:
    if value == 0:
        return 1
    # Exact count with correction, no conversion to decimal required.
    n = -value if value < 0 else value
    bits = n.bit_length()
    estimate = max(1, int((bits - 1) * math.log10(2)) + 1)
    p = 10 ** (estimate - 1)
    if n < p:
        return estimate - 1
    if n >= p * 10:
        return estimate + 1
    return estimate
