# =============================================================================
# MathKernel - Numba-compiled array kernels for exact small-integer modular arithmetic
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Numba-compiled array kernels for exact small-integer modular arithmetic.

These are building blocks for sieve-style scan loops (e.g. quadratic-residue
batteries), where millions of modular tests run inside one compiled loop with
no per-operation Python overhead. They are deliberately array-in/array-out:
the dict-shaped integer_batch API is glue-bound, and routing it through here
measured slower than the existing C-level bigint path.

Exactness fragments:

* 32-bit (powmod32, is_prime32, modinv32): residues < m < 2^32, so every
  product fits in uint64 and hardware division is exact. is_prime32 uses
  deterministic Miller-Rabin with bases {2, 7, 61} (valid below 2^32).
* 63-bit (mod, gcd): no products are formed, so hardware uint64 operations
  are exact for |operands| < 2^63.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

if HAVE_NUMBA:

    @njit(cache=True)
    def _powmod32(base, exp, m):
        # base, m < 2^32 so residue products fit uint64 exactly.
        if m == 1:
            return np.uint64(0)
        r = np.uint64(1)
        b = base % m
        e = exp
        while e > 0:
            if e & 1:
                r = (r * b) % m
            b = (b * b) % m
            e >>= 1
        return r

    @njit(cache=True)
    def _is_prime32(n):
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n == p:
                return True
            if n % p == 0:
                return False
        nu = np.uint64(n)
        nm1 = nu - 1
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        du = np.uint64(d)
        for a in (2, 7, 61):  # deterministic for n < 2^32
            av = np.uint64(a) % nu
            if av == 0:
                continue
            x = _powmod32(av, du, nu)
            if x == 1 or x == nm1:
                continue
            composite = True
            for _ in range(r - 1):
                x = (x * x) % nu
                if x == nm1:
                    composite = False
                    break
            if composite:
                return False
        return True

    @njit(cache=True)
    def _modinv32(a, m):
        # Inverse in [0, m), or -1 when gcd(a, m) != 1.
        # q * newt < 2^32 * 2^32 fits uint64: exact without 128-bit mulmod.
        if m == 1:
            return np.int64(0)
        a = a % m
        t = np.int64(0)
        newt = np.int64(1)
        r = m
        newr = a
        while newr != 0:
            q = r // newr
            r, newr = newr, r - q * newr
            v = np.int64((np.uint64(q) * np.uint64(newt)) % np.uint64(m))
            tv = t - v
            if tv < 0:
                tv += m
            t, newt = newt, tv
        if r != 1:
            return np.int64(-1)
        return t

    @njit(parallel=True, cache=True)
    def _powmod_kernel(bases, exps, mods, out):
        for i in prange(out.shape[0]):
            out[i] = _powmod32(np.uint64(bases[i]), np.uint64(exps[i]),
                               np.uint64(mods[i]))

    @njit(parallel=True, cache=True)
    def _is_prime_kernel(ns, out):
        for i in prange(out.shape[0]):
            out[i] = 1 if _is_prime32(ns[i]) else 0

    @njit(parallel=True, cache=True)
    def _modinv_kernel(vals, mods, out):
        for i in prange(out.shape[0]):
            out[i] = _modinv32(vals[i], mods[i])

    @njit(parallel=True, cache=True)
    def _mod_kernel(vals, mods, out):
        for i in prange(out.shape[0]):
            out[i] = vals[i] % mods[i]

    @njit(parallel=True, cache=True)
    def _gcd_kernel(vals, out):
        for i in prange(out.shape[0]):
            g = np.uint64(0)
            for j in range(vals.shape[1]):
                b = np.uint64(abs(vals[i, j]))
                while b != 0:
                    g, b = b, g % b
            out[i] = np.int64(g)


def _need_numba() -> None:
    if not HAVE_NUMBA:
        raise RuntimeError("numba is not available")


def powmod_array(bases: np.ndarray, exps: np.ndarray, mods: np.ndarray) -> np.ndarray:
    """bases[i] ** exps[i] % mods[i], exact for bases/mods < 2^32."""
    _need_numba()
    out = np.empty(len(bases), np.uint64)
    _powmod_kernel(np.asarray(bases, np.uint64), np.asarray(exps, np.uint64),
                   np.asarray(mods, np.uint64), out)
    return out


def is_prime_array(ns: np.ndarray) -> np.ndarray:
    """Deterministic primality for 0 <= n < 2^32, uint8 out."""
    _need_numba()
    out = np.empty(len(ns), np.uint8)
    _is_prime_kernel(np.asarray(ns, np.int64), out)
    return out


def modinv_array(vals: np.ndarray, mods: np.ndarray) -> np.ndarray:
    """Modular inverse in [0, m), or -1 when not invertible. vals/mods < 2^32."""
    _need_numba()
    out = np.empty(len(vals), np.int64)
    _modinv_kernel(np.asarray(vals, np.int64), np.asarray(mods, np.int64), out)
    return out


def mod_array(vals: np.ndarray, mods: np.ndarray) -> np.ndarray:
    """vals[i] % mods[i], exact for 0 <= vals, 0 < mods < 2^63."""
    _need_numba()
    out = np.empty(len(vals), np.int64)
    _mod_kernel(np.asarray(vals, np.int64), np.asarray(mods, np.int64), out)
    return out


def gcd_array(vals: np.ndarray) -> np.ndarray:
    """Row-wise gcd of a rectangular int64 array (0-pad ragged rows)."""
    _need_numba()
    out = np.empty(vals.shape[0], np.int64)
    _gcd_kernel(np.asarray(vals, np.int64), out)
    return out
