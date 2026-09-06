# =============================================================================
# MathKernel - Exact binary-field arithmetic: GF(2^m) in polynomial basis
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Exact binary-field arithmetic: GF(2^m) in polynomial basis.

Elements are integers whose bit i is the coefficient of x^i. The field
modulus is x^m + red with red < 2^m. Irreducibility is verified with Rabin's
test at construction time — without it the structure is a ring, not a field,
and inverses/roots would be unsound.

Also provides the LFSR-style construction used by the xoroshiro analysis
pipeline: given the columns of a GF(2)-linear transition on m bits, build the
dual-orbit cyclic basis and recover the transition's minimal polynomial as
the field modulus.
"""
from __future__ import annotations

from .gf2m_fast import fast_ops


def _lowest_bit(v: int) -> int:
    return (v & -v).bit_length() - 1


class GF2mField:
    """GF(2^m) with modulus x^m + red.

    For m <= 128 with numba installed, mul/power dispatch to compiled
    two-limb kernels (gf2m_fast); results are bit-identical to the
    pure-Python path. Set allow_fast=False to force the reference path.
    """

    def __init__(self, m: int, red: int, *, allow_fast: bool = True):
        if m < 2 or m > 1024:
            raise ValueError("degree m must be between 2 and 1024")
        if red < 0 or red >= (1 << m):
            raise ValueError("red must be in [0, 2^m)")
        self.m = m
        self.red = red
        self.mask = (1 << m) - 1
        self._fast = fast_ops(m, red) if allow_fast else None

    # -- field arithmetic ---------------------------------------------------

    def add(self, a: int, b: int) -> int:
        return a ^ b

    def mul(self, a: int, b: int) -> int:
        if self._fast is not None:
            return self._fast.mul(a, b)
        out = 0
        while b:
            if b & 1:
                out ^= a
            b >>= 1
            carry = a >> (self.m - 1)
            a = (a << 1) & self.mask
            if carry:
                a ^= self.red
        return out

    def power(self, a: int, exponent: int) -> int:
        if exponent < 0:
            raise ValueError("exponent must be non-negative")
        if a == 0:
            return 1 if exponent == 0 else 0
        # Every nonzero element satisfies a^(2^m - 1) = 1, so exponents may
        # be reduced mod 2^m - 1 (this also keeps the fast path applicable
        # for huge exponents such as jump indices).
        exponent %= (1 << self.m) - 1
        if self._fast is not None:
            return self._fast.pow(a, exponent)
        out = 1
        while exponent:
            if exponent & 1:
                out = self.mul(out, a)
            a = self.mul(a, a)
            exponent >>= 1
        return out

    def inv(self, a: int) -> int:
        if not a:
            raise ZeroDivisionError("0 has no inverse in GF(2^m)")
        return self.power(a, (1 << self.m) - 2)

    def div(self, a: int, b: int) -> int:
        return self.mul(a, self.inv(b))

    def sqrt(self, a: int) -> int:
        return self.power(a, 1 << (self.m - 1))

    def trace(self, a: int) -> int:
        total = 0
        for _ in range(self.m):
            total ^= a
            a = self.mul(a, a)
        return total

    def quadratic_roots(self, c0: int, c1: int, c2: int) -> list[int]:
        """Solve c0 + c1*r + c2*r^2 = 0 in GF(2^m)."""
        if c2 == 0:
            return [] if c1 == 0 else [self.mul(c0, self.inv(c1))]
        a, b = self.div(c1, c2), self.div(c0, c2)
        if a == 0:
            return [self.sqrt(b)]
        target = self.div(b, self.mul(a, a))
        # Solve z^2 + z = target; the map z -> z^2 + z is GF(2)-linear.
        basis: list[int | None] = [None] * self.m
        coeff = [0] * self.m
        for i in range(self.m):
            e = 1 << i
            value, c = self.mul(e, e) ^ e, e
            while value:
                p = value.bit_length() - 1
                if basis[p] is None:
                    basis[p], coeff[p] = value, c
                    break
                value ^= basis[p]  # type: ignore[operator]
                c ^= coeff[p]
        value, solution = target, 0
        while value:
            p = value.bit_length() - 1
            if basis[p] is None:
                return []  # target has trace 1; no solution
            value ^= basis[p]  # type: ignore[operator]
            solution ^= coeff[p]
        root = self.mul(a, solution)
        return [root, root ^ a]

    # -- irreducibility (Rabin) ----------------------------------------------

    def _poly_gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, self._poly_mod(a, b)
        return a

    @staticmethod
    def _poly_mod(a: int, b: int) -> int:
        db = b.bit_length() - 1
        while a.bit_length() - 1 >= db:
            a ^= b << (a.bit_length() - 1 - db)
        return a

    def verify_irreducible(self) -> bool:
        """Rabin's test for f = x^m + red over GF(2)."""
        f = (1 << self.m) | self.red
        # x^(2^(m/q)) for each prime divisor q of m; here generic trial division.
        n = self.m
        primes: set[int] = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                primes.add(d)
                n //= d
            d += 1
        if n > 1:
            primes.add(n)
        x = 2
        for q in sorted(primes):
            h = x
            for _ in range(self.m // q):
                h = self.mul(h, h)
            if self._poly_gcd(h ^ x, f) != 1:
                return False
        h = x
        for _ in range(self.m):
            h = self.mul(h, h)
        return h == x


# ---------------------------------------------------------------------------
# Field construction from a GF(2)-linear transition
# ---------------------------------------------------------------------------

def rows_from_cols(cols: list[int]) -> list[int]:
    """Transpose a column-major bit-matrix into row parity masks."""
    rows = [0] * len(cols)
    for j, col in enumerate(cols):
        while col:
            bit = _lowest_bit(col)
            rows[bit] |= 1 << j
            col &= col - 1
    return rows


def cols_from_rows(rows: list[int], width: int) -> list[int]:
    """Transpose row parity masks back into column-major form."""
    cols = [0] * width
    for i, row in enumerate(rows):
        while row:
            bit = _lowest_bit(row)
            cols[bit] |= 1 << i
            row &= row - 1
    return cols


# ---------------------------------------------------------------------------
# GF(2) matrix algebra (row-major bit-packed integers)
# ---------------------------------------------------------------------------

def gf2_rank(rows: list[int], width: int) -> tuple[int, int]:
    """(rank, nullity) of the row space; nullity = width - rank."""
    work = [r for r in rows if r]
    rank = 0
    for col in range(width - 1, -1, -1):
        pivot = next((i for i in range(rank, len(work)) if (work[i] >> col) & 1), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for i in range(len(work)):
            if i != rank and (work[i] >> col) & 1:
                work[i] ^= work[rank]
        rank += 1
        if rank == len(work):
            break
    return rank, width - rank


def gf2_nullspace(rows: list[int], width: int) -> list[int]:
    """Basis of the right nullspace {v : parity(row & v) = 0 for all rows}."""
    work = [r for r in rows if r]
    pivot_of_row: list[int] = []
    rank = 0
    for col in range(width - 1, -1, -1):
        pivot = next((i for i in range(rank, len(work)) if (work[i] >> col) & 1), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        for i in range(len(work)):
            if i != rank and (work[i] >> col) & 1:
                work[i] ^= work[rank]
        pivot_of_row.append(col)
        rank += 1
        if rank == len(work):
            break
    pivot_cols = set(pivot_of_row)
    basis = []
    for free in range(width):
        if free in pivot_cols:
            continue
        v = 1 << free
        for row, pcol in zip(work, pivot_of_row):
            if (row >> free) & 1:
                v |= 1 << pcol
        basis.append(v)
    return basis


def gf2_left_nullspace(rows: list[int], width: int) -> list[int]:
    """Basis of the left nullspace {v : v * M = 0} (v as row vector)."""
    return gf2_nullspace(cols_from_rows(rows, width), len(rows))


def carryfree_cols(c: int, width: int) -> list[int]:
    """Columns of the carry-free multiply-by-c map on width-bit words.

    P_c(x) = XOR over set bits j of c of (x << j), truncated to width bits.
    This is the GF(2)-linear analogue of integer multiplication by c.
    """
    mask = (1 << width) - 1
    return [(c << j) & mask for j in range(width)]


def gf2_apply(rows: list[int], v: int, width: int) -> int:
    """Matrix-vector product M*v with M given as row parity masks."""
    out = 0
    for i in range(min(len(rows), width)):
        if (rows[i] & v).bit_count() & 1:
            out |= 1 << i
    return out


def gf2_apply_transpose(rows: list[int], v: int, width: int) -> int:
    """Matrix-vector product M^T*v: XOR of the ROWS of M selected by v's bits
    (row j of M is column j of M^T)."""
    out = 0
    while v:
        bit = _lowest_bit(v)
        if bit < len(rows):
            out ^= rows[bit]
        v &= v - 1
    return out & ((1 << width) - 1)


def gf2_rows_mul(a_rows: list[int], b_rows: list[int], width: int) -> list[int]:
    """Product A*B of square row-packed matrices: row i of A*B is the XOR of
    rows of B selected by the set bits of A's row i."""
    out = []
    for row in a_rows:
        acc = 0
        r = row
        while r:
            bit = _lowest_bit(r)
            if bit < len(b_rows):
                acc ^= b_rows[bit]
            r &= r - 1
        out.append(acc & ((1 << width) - 1))
    return out


def gf2_rows_power(rows: list[int], k: int, width: int) -> list[int]:
    """M^k for a row-packed square matrix by square-and-multiply."""
    if k < 0:
        raise ValueError("matrix power must be non-negative")
    result = [1 << i for i in range(width)]
    base = list(rows)
    while k:
        if k & 1:
            result = gf2_rows_mul(result, base, width)
        base = gf2_rows_mul(base, base, width)
        k >>= 1
    return result


_BYTE_REV = bytes(int(f"{i:08b}"[::-1], 2) for i in range(256))


def _reverse_bits(value: int, n: int) -> int:
    """Bit-reverse the low n bits of value (bit i <-> bit n-1-i)."""
    nbytes = (n + 7) // 8
    raw = value.to_bytes(nbytes, "little")
    rev = int.from_bytes(raw.translate(_BYTE_REV)[::-1], "little")
    return rev >> (8 * nbytes - n)


def gf2_berlekamp_massey(bits: int, n: int) -> int:
    """Connection polynomial of a GF(2) bit sequence via Berlekamp-Massey.

    `bits` packs the sequence s_0..s_{n-1} with s_i at bit position i.
    Returns the connection polynomial C(x) = 1 + c_1 x + ... + c_L x^L as an
    integer (bit j = coefficient of x^j), i.e. the minimal recurrence
    s_n = sum_{j=1..L} c_j s_{n-j} satisfied from position L onward.

    Bit-packed: the discrepancy is one bigint AND + popcount per step, so
    recovering a degree-~20k polynomial from 40k bits takes well under a
    second; a compiled kernel would not pay for its JIT warmup here.
    """
    if n < 1:
        raise ValueError("need at least one bit")
    if bits >> n:
        raise ValueError("bits value exceeds n bits")
    rev = _reverse_bits(bits, n)  # rev bit i = s_{n-1-i}: parity(C & rev>>(n-1-k)) = sum_j C_j s_{k-j}
    conn, basis, deg, gap = 1, 1, 0, 1
    for k in range(n):
        d = (conn & (rev >> (n - 1 - k))).bit_count() & 1
        if d:
            prev = conn
            conn ^= basis << gap
            if 2 * deg <= k:
                deg = k + 1 - deg
                basis = prev
                gap = 1
            else:
                gap += 1
        else:
            gap += 1
    return conn


def gf2_sequence_satisfies(bits: int, n: int, conn: int, start: int | None = None) -> int:
    """Count recurrence violations of `conn` on s_start..s_{n-1}; 0 = exact."""
    deg = conn.bit_length() - 1
    i0 = deg if start is None else max(start, deg)
    if i0 >= n:
        return 0
    rev = _reverse_bits(bits, n)
    violations = 0
    for k in range(i0, n):
        violations += (conn & (rev >> (n - 1 - k))).bit_count() & 1
    return violations


class TransitionField:
    """GF(2^m) built from the dual orbit of a linear transition matrix.

    columns[j] is the image of the j-th unit vector under the transition.
    The dual orbit e0, e0*M, e0*M^2, ... must span the whole space (cyclic);
    the coordinates of e0*M^m in that basis give the minimal polynomial,
    which becomes the field modulus after irreducibility verification.
    """

    def __init__(self, columns: list[int]):
        self.m = len(columns)
        if self.m < 2:
            raise ValueError("need at least 2 transition columns")
        self.rows = rows_from_cols(columns)
        orbit, row = [], 1
        for _ in range(self.m + 1):
            orbit.append(row)
            out = 0
            while row:
                out ^= self.rows[_lowest_bit(row)]
                row &= row - 1
            row = out
        basis: list[int | None] = [None] * self.m
        coeff = [0] * self.m
        for i, value in enumerate(orbit[: self.m]):
            c = 1 << i
            while value:
                p = value.bit_length() - 1
                if basis[p] is None:
                    basis[p], coeff[p] = value, c
                    break
                value ^= basis[p]  # type: ignore[operator]
                c ^= coeff[p]
        if any(x is None for x in basis):
            raise ValueError("transition dual orbit is not cyclic; no field construction")
        self.basis, self.coeff, self.orbit = basis, coeff, orbit
        red = self.coords(orbit[self.m])
        self.field = GF2mField(self.m, red)
        if not self.field.verify_irreducible():
            raise ValueError(
                f"minimal polynomial x^{self.m} + {red:0{self.m // 4}x} is reducible; "
                "the construction yields a ring, not a field")

    @property
    def red(self) -> int:
        return self.field.red

    def coords(self, value: int) -> int:
        out = 0
        while value:
            p = value.bit_length() - 1
            value ^= self.basis[p]  # type: ignore[operator]
            out ^= self.coeff[p]
        return out

    def row_from_coords(self, coordinates: int) -> int:
        row = 0
        while coordinates:
            low = coordinates & -coordinates
            row ^= self.orbit[_lowest_bit(low)]
            coordinates ^= low
        return row

    def root_jump_rows(self, root: int) -> list[int]:
        """Rows of the state-jump action induced by a field element."""
        return [
            self.row_from_coords(self.field.mul(self.coords(1 << bit), root))
            for bit in range(self.m)
        ]

    def jump_rows(self, k: int) -> list[int]:
        """Rows of the K-step jump T^K (= action of alpha^K)."""
        return self.root_jump_rows(self.field.power(2, k))

    @staticmethod
    def apply_rows(rows: list[int], state: int) -> int:
        out = 0
        for bit, row in enumerate(rows):
            out |= ((row & state).bit_count() & 1) << bit
        return out

    def closure_roots(self, row_sets: list[list[int]]) -> list[dict]:
        """Solve the closure equation for each set of state rows.

        Two rows give the linear equation c0 + c1*r = 0; three rows give the
        quadratic c0 + c1*r + c2*r^2 = 0. Roots 0 and 1 are trivial and
        excluded, matching the reference pipeline.
        """
        out = []
        for rows in row_sets:
            coeff = [self.coords(r) for r in rows]
            if len(coeff) == 3:
                roots = self.field.quadratic_roots(*coeff)
            elif len(coeff) == 2 and coeff[1]:
                roots = [self.field.div(coeff[0], coeff[1])]
            else:
                roots = []
            roots = [r for r in roots if r not in (0, 1)]
            residuals = []
            for root in roots:
                value = coeff[0] ^ self.field.mul(coeff[1], root)
                if len(coeff) == 3:
                    value ^= self.field.mul(coeff[2], self.field.mul(root, root))
                residuals.append(value)
            out.append({"coefficients": coeff, "roots": roots, "residuals": residuals})
        return out
