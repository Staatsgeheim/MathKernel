# =============================================================================
# MathKernel - Tests for the GF(2^m) engine and the xoroshiro128** closure reproduction."""
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Tests for the GF(2^m) engine and the xoroshiro128** closure reproduction."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from mathkernel.gf2m import (
    GF2mField,
    TransitionField,
    carryfree_cols,
    gf2_left_nullspace,
    gf2_nullspace,
    gf2_rank,
    rows_from_cols,
)
from mathkernel.gf2m_fast import HAVE_NUMBA
from mathkernel.kernel import MathKernel
from mathkernel.settings import Settings
from mathkernel.transforms import _fwht_python, fwht

MASK64 = (1 << 64) - 1
REFERENCE = (Path(__file__).resolve().parents[1] / "scripts" / "data"
             / "xoroshiro128starstar_closure_roots.json")

# xoroshiro128** 1.0 (24,16,37) specification (input to the math, not the math)
POSITIONS = [(0, b) for b in (57, 56, 55, 54, 53, 52, 51, 50, 49)]


def rotl(x: int, k: int) -> int:
    return ((x << k) & MASK64) | (x >> (64 - k))


def xoro_step(state: int) -> int:
    s0, s1 = state & MASK64, state >> 64
    x = s1 ^ s0
    return (rotl(s0, 24) ^ x ^ ((x << 16) & MASK64)) | (rotl(x, 37) << 64)


def mask_rows(mask: int) -> list[int]:
    rows = [0, 0]
    for i in range(2 * len(POSITIONS)):
        if (mask >> i) & 1:
            lag, variable = divmod(i, len(POSITIONS))
            word, bit = POSITIONS[variable]
            rows[lag] ^= 1 << (bit + 64 * word)
    return rows


# ---------------------------------------------------------------------------
# Small-field unit tests
# ---------------------------------------------------------------------------

def test_gf16_known_products():
    f = GF2mField(4, 0b0011)  # x^4 + x + 1
    assert f.mul(2, 2) == 4
    assert f.mul(2, 8) == 3       # x * x^3 = x^4 = x + 1
    assert f.mul(3, 3) == 5       # (x+1)^2 = x^2 + 1
    assert f.add(0b1010, 0b0110) == 0b1100


def test_gf16_inverse_and_sqrt():
    f = GF2mField(4, 0b0011)
    for a in range(1, 16):
        assert f.mul(a, f.inv(a)) == 1
        assert f.mul(f.sqrt(a), f.sqrt(a)) == a


def test_gf16_trace():
    f = GF2mField(4, 0b0011)
    assert f.trace(0) == 0
    assert f.trace(1) == 0  # m=4 even: 1+1+1+1 = 0
    values = {f.trace(a) for a in range(16)}
    assert values == {0, 1}


def test_gf16_quadratic_roots():
    f = GF2mField(4, 0b0011)
    # z^2 + z = 0 -> z in {0, 1}: c0=0, c1=1, c2=1
    assert sorted(f.quadratic_roots(0, 1, 1)) == [0, 1]
    # linear fallback: 1 + 3*r = 0 -> r = 1/3
    r = f.quadratic_roots(1, 3, 0)
    assert len(r) == 1 and f.mul(3, r[0]) == 1
    # pure square: 5 + 0*r + r^2 = 0 -> r = sqrt(5)
    r = f.quadratic_roots(5, 0, 1)
    assert len(r) == 1 and f.mul(r[0], r[0]) == 5
    # every solvable quadratic returns actual roots
    rng = random.Random(7)
    for _ in range(20):
        c0, c1, c2 = (rng.randrange(16) for _ in range(3))
        for root in f.quadratic_roots(c0, c1, c2):
            assert c0 ^ f.mul(c1, root) ^ f.mul(c2, f.mul(root, root)) == 0


def test_gf256_field_axioms():
    f = GF2mField(8, 0x1B)  # AES polynomial x^8+x^4+x^3+x+1
    rng = random.Random(11)
    for _ in range(50):
        a, b, c = (rng.randrange(256) for _ in range(3))
        assert f.add(a, b) == a ^ b
        assert f.mul(a, f.add(b, c)) == f.add(f.mul(a, b), f.mul(a, c))
        assert f.mul(a, b) == f.mul(b, a)
        if a:
            assert f.mul(a, f.inv(a)) == 1
            assert f.div(f.mul(a, b), a) == b


def test_reducible_modulus_rejected():
    kernel = MathKernel()
    # x^4 + x^2 + 1 = (x^2 + x + 1)^2 over GF(2)
    result = kernel.gf2m_create(4, "5")
    assert not result.ok
    assert "reducible" in result.errors[0]


def test_irreducibility_self_check():
    assert GF2mField(4, 0b0011).verify_irreducible()
    assert not GF2mField(4, 0b0101).verify_irreducible()
    assert GF2mField(8, 0x1B).verify_irreducible()


def test_kernel_create_and_compute():
    kernel = MathKernel()
    created = kernel.gf2m_create(4, "3")
    assert created.ok and created.data["irreducible"]
    fid = created.data["field_id"]
    mul = kernel.gf2m_compute(fid, "mul", ["2", "8"])
    assert mul.ok and mul.data["result"] == "3"
    inv = kernel.gf2m_compute(fid, "inv", ["2"])
    assert inv.ok
    check = kernel.gf2m_compute(fid, "mul", ["2", inv.data["result"]])
    assert check.data["result"] == "1"
    roots = kernel.gf2m_compute(fid, "quadratic_roots", ["0", "1", "1"])
    assert roots.data["roots"] == ["0", "1"]


def test_kernel_errors():
    kernel = MathKernel()
    assert not kernel.gf2m_compute("gf2m_nonexistent", "mul", ["1", "1"]).ok
    created = kernel.gf2m_create(4, "3")
    fid = created.data["field_id"]
    assert not kernel.gf2m_compute(fid, "mul", ["1"]).ok           # arity
    assert not kernel.gf2m_compute(fid, "mul", ["zz", "1"]).ok     # not hex
    assert not kernel.gf2m_compute(fid, "mul", ["10", "1"]).ok     # too wide for m=4
    assert not kernel.gf2m_compute(fid, "inv", ["0"]).ok           # 0 has no inverse
    assert not kernel.gf2m_compute(fid, "wat", ["1"]).ok           # unknown op
    # coords requires a transition-derived field
    assert not kernel.gf2m_coords(fid, "1").ok


def test_transition_field_small():
    # Transition = multiplication by x in GF(2^4)/(x^4+x+1): columns are x*e_i.
    columns = [2, 4, 8, 3]
    tf = TransitionField(columns)
    assert tf.m == 4 and tf.red == 0b0011
    # root 2 (= x) must induce exactly the transition itself
    assert tf.root_jump_rows(2) == rows_from_cols(columns)


def test_capabilities_advertise_gf2m():
    kernel = MathKernel()
    caps = kernel.capabilities()
    assert "gf2m_create" in caps["operations"]
    assert caps["gf2m"]["max_degree"] == 1024


# ---------------------------------------------------------------------------
# Compiled fast path vs pure-Python reference (differential)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_path_matches_reference_gf256():
    fast = GF2mField(8, 0x1B)
    slow = GF2mField(8, 0x1B, allow_fast=False)
    assert fast._fast is not None and slow._fast is None
    rng = random.Random(23)
    for _ in range(200):
        a, b = rng.randrange(256), rng.randrange(256)
        e = rng.randrange(1 << 16)
        assert fast.mul(a, b) == slow.mul(a, b)
        assert fast.power(a, e) == slow.power(a, e)
        if a:
            assert fast.inv(a) == slow.inv(a)
        assert fast.sqrt(a) == slow.sqrt(a)
        assert fast.quadratic_roots(a, b, 5) == slow.quadratic_roots(a, b, 5)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_path_matches_reference_gf128():
    red = int("0008828e513b43d5095b8f76579aa001", 16)
    fast = GF2mField(128, red)
    slow = GF2mField(128, red, allow_fast=False)
    assert fast._fast is not None
    rng = random.Random(29)
    for _ in range(20):
        a, b = rng.getrandbits(128), rng.getrandbits(128)
        assert fast.mul(a, b) == slow.mul(a, b)
        assert fast.power(a, (1 << 128) - 2) == slow.inv(a)  # inv via fast pow
        assert fast.sqrt(a) == slow.sqrt(a)
        c0, c1, c2 = rng.getrandbits(128), rng.getrandbits(128), rng.getrandbits(128)
        assert fast.quadratic_roots(c0, c1, c2) == slow.quadratic_roots(c0, c1, c2)
    assert fast.verify_irreducible() == slow.verify_irreducible()


XORWOW_RED = int("00000f0e0f3c0035000621210861003000060001", 16)
XS1024_RED = int(
    "0000000000007879787878786d3815400440024003007b2853116c08605c805fa1422cb7814f5c68"
    "0040f0e46e848800cd40a7e2537771eababab341e2554b59df6a7cadba32bca9e860f083d70158"
    "c634b3216457d7b0284a32d044029b08f7030d5352015561300111e1c02bc181802200aa001400"
    "f0001000000000000001", 16)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_path_matches_reference_gf160():
    fast = GF2mField(160, XORWOW_RED)
    slow = GF2mField(160, XORWOW_RED, allow_fast=False)
    assert fast._fast is not None
    rng = random.Random(41)
    for _ in range(10):
        a, b = rng.getrandbits(160), rng.getrandbits(160)
        assert fast.mul(a, b) == slow.mul(a, b)
        assert fast.power(a, (1 << 160) - 2) == slow.inv(a)
        c0, c1, c2 = (rng.getrandbits(160) for _ in range(3))
        assert fast.quadratic_roots(c0, c1, c2) == slow.quadratic_roots(c0, c1, c2)


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_path_matches_reference_gf1024():
    fast = GF2mField(1024, XS1024_RED)
    slow = GF2mField(1024, XS1024_RED, allow_fast=False)
    assert fast._fast is not None
    rng = random.Random(43)
    for _ in range(3):
        a, b = rng.getrandbits(1024), rng.getrandbits(1024)
        assert fast.mul(a, b) == slow.mul(a, b)
        assert fast.power(a, (1 << 1024) - 2) == slow.inv(a)


def test_power_exponent_reduction():
    # a^(2^m - 1) = 1 for nonzero a; huge exponents reduce mod 2^m - 1
    field = GF2mField(128, int("0008828e513b43d5095b8f76579aa001", 16))
    a = 0xDEADBEEFCAFEBABE1234567890ABCDEF
    assert field.power(a, (1 << 128) - 1) == 1
    assert field.power(a, (1 << 128)) == a
    assert field.power(0, 5) == 0
    assert field.power(0, 0) == 1
    k = 91801091522189471493542482995818735741  # huge jump index
    assert field.power(2, k) == field.power(2, k % ((1 << 128) - 1))


# ---------------------------------------------------------------------------
# GF(2) matrix algebra and jump rows
# ---------------------------------------------------------------------------

def test_gf2_rank_and_nullity():
    rows = [0b1100, 0b1010, 0b0110]  # row2 = row0 ^ row1 -> rank 2
    assert gf2_rank(rows, 4) == (2, 2)
    assert gf2_rank([], 4) == (0, 4)
    assert gf2_rank([0b0001, 0b0010, 0b0100, 0b1000], 4) == (4, 0)


def test_gf2_nullspace_roundtrip():
    rng = random.Random(47)
    rows = [rng.getrandbits(16) for _ in range(10)]
    basis = gf2_nullspace(rows, 16)
    _rank, nullity = gf2_rank(rows, 16)
    assert len(basis) == nullity
    for v in basis:
        for row in rows:
            assert ((row & v).bit_count() & 1) == 0
    # left nullspace annihilates from the left
    left = gf2_left_nullspace(rows, 16)
    for v in left:
        acc = 0
        for i, row in enumerate(rows):
            if (v >> i) & 1:
                acc ^= row
        assert acc == 0


def test_carryfree_cols():
    # P_3(x) = x ^ (x << 1) on 4-bit words
    cols = carryfree_cols(3, 4)
    assert cols[0] == 0b0011
    assert cols[1] == 0b0110
    assert cols[3] == 0b1000  # (3 << 3) truncated to 4 bits
    # apply via columns: P_3(0b0101) = 0b0101 ^ 0b1010 = 0b1111
    v = 0b0101
    out = 0
    while v:
        low = v & -v
        out ^= cols[low.bit_length() - 1]
        v &= v - 1
    assert out == 0b1111


def test_jump_rows_composition():
    tf = TransitionField([xoro_step(1 << i) for i in range(128)])
    a, b = 12345, 67890
    ra, rb, rab = tf.jump_rows(a), tf.jump_rows(b), tf.jump_rows(a + b)
    probe = 0x123456789ABCDEF0 | (0x0F0E0D0C0B0A0908 << 64)
    assert TransitionField.apply_rows(rb, TransitionField.apply_rows(ra, probe)) == \
        TransitionField.apply_rows(rab, probe)


def test_kernel_gf2_tools():
    kernel = MathKernel()
    rank = kernel.gf2_rank(["c", "a", "6"], 4)
    assert rank.ok and rank.data == {"rank": 2, "nullity": 2, "width": 4}
    right = kernel.gf2_nullspace(["c", "a", "6"], 4, "right")
    assert right.ok and right.data["dimension"] == 2
    left = kernel.gf2_nullspace(["c", "a", "6"], 4, "left")
    assert left.ok and left.data["dimension"] == 1  # 3 rows, rank 2
    assert not kernel.gf2_nullspace(["c"], 4, "sideways").ok
    cols = kernel.gf2_carryfree_cols("3", 4)
    assert cols.ok and cols.data["columns"] == ["3", "6", "c", "8"]
    assert not kernel.gf2_carryfree_cols("0", 4).ok


def test_kernel_gf2m_jump_rows():
    kernel = MathKernel()
    field = kernel.gf2m_from_transition([f"{xoro_step(1 << i):032x}" for i in range(128)])
    assert field.ok
    field_id = field.data["field_id"]
    rows1 = [int(r, 16) for r in kernel.gf2m_jump_rows(field_id, "1").data["rows"]]
    probe = 0xABCDEF1234567890 | (0x1234567890ABCDEF << 64)
    assert TransitionField.apply_rows(rows1, probe) == xoro_step(probe)
    # jump rows via alpha^k match root_jump_rows(power(2, k))
    rows_k = kernel.gf2m_jump_rows(field_id, "12345").data["rows"]
    alpha_k = kernel.gf2m_compute(field_id, "pow", ["2"], exponent=f"{12345:x}").data["result"]
    rows_r = kernel.gf2m_root_jump_rows(field_id, alpha_k).data["rows"]
    assert rows_k == rows_r
    assert not kernel.gf2m_jump_rows(field_id, "-1").ok


# ---------------------------------------------------------------------------
# FWHT
# ---------------------------------------------------------------------------

def test_fwht_known_values():
    # delta at 0 -> all ones; delta at 1 -> (-1)^popcount(mask)
    assert fwht([1, 0, 0, 0]) == [1, 1, 1, 1]
    assert fwht([0, 1, 0, 0]) == [1, -1, 1, -1]
    assert fwht([1, 1]) == [2, 0]


def test_fwht_involution():
    rng = random.Random(31)
    values = [rng.randrange(-1000, 1000) for _ in range(256)]
    assert fwht(fwht(values)) == [256 * v for v in values]


def test_fwht_fast_matches_reference():
    rng = random.Random(37)
    values = [rng.randrange(-10**6, 10**6) for _ in range(1024)]
    assert fwht(values) == _fwht_python(values)


def test_fwht_big_integer_fallback():
    # sum(abs) exceeds the int64 safe bound -> arbitrary-precision path
    big = 1 << 70
    values = [big, -big, big, big]
    assert fwht(values) == _fwht_python(values) == [2 * big, 2 * big, -2 * big, 2 * big]


def test_fwht_rejects_non_power_of_two():
    kernel = MathKernel()
    assert not kernel.fwht(["1", "2", "3"]).ok
    assert not kernel.fwht([]).ok


def test_kernel_fwht_roundtrip():
    kernel = MathKernel()
    result = kernel.fwht(["1", "0", "0", "0"])
    assert result.ok
    assert result.data["values"] == ["1", "1", "1", "1"]
    assert result.data["variables"] == 2
    assert result.trust.value == "exact"


# ---------------------------------------------------------------------------
# End-to-end pipeline from scratch, self-certifying (no reference artifacts)
# ---------------------------------------------------------------------------

def test_xoroshiro128starstar_end_to_end_self_certifying():
    # bulk spectral pipeline: programmatic consumer, raise the output budget
    kernel = MathKernel(Settings(max_output_size_bytes=256_000_000))
    n_samples, n_validate = 20_000, 4_096

    # published reference vector (spec conformance)
    state, got = 1 | (2 << 64), []
    published = [5760, 97769243520, 9706862127477703552, 9223447511460779954,
                 8358291023205304566, 15695619998649302768, 8517900938696309774,
                 16586480348202605369, 6959129367028440372, 16822147227405758281]
    for _ in published:
        got.append((rotl(((state & MASK64) * 5) & MASK64, 7) * 9) & MASK64)
        state = xoro_step(state)
    assert got == published

    # fresh samples -> carry-difference observations -> exact Walsh spectrum
    rng = random.Random(0x5EED_10)
    accum = [0] * (1 << 18)
    for _ in range(n_samples):
        s0, s1 = rng.getrandbits(128), rng.getrandbits(128)
        x = 0
        for i, (word, bit) in enumerate(POSITIONS):
            x |= ((s0 >> (bit + 64 * word)) & 1) << i
            x |= ((s1 >> (bit + 64 * word)) & 1) << (i + 9)
        out0 = (rotl(((s0 & MASK64) * 5) & MASK64, 7) * 9) & MASK64
        out1 = (rotl(((s1 & MASK64) * 5) & MASK64, 7) * 9) & MASK64
        accum[x] += 1 if ((out0 - out1) & MASK64) >> 63 == 0 else -1
    spectrum = kernel.fwht([str(v) for v in accum])
    assert spectrum.ok
    coeffs = [int(v) for v in spectrum.data["values"]]
    ranked = sorted(((m, s / n_samples) for m, s in enumerate(coeffs) if m),
                    key=lambda t: (-abs(t[1]), t[0]))
    strong = [(m, r) for m, r in ranked[:24] if abs(r) >= 0.06]
    assert len(strong) >= 3, "expected several masks above the noise floor"

    # closure roots via the kernel, from the transition alone
    columns = [xoro_step(1 << i) for i in range(128)]
    field = kernel.gf2m_from_transition([f"{c:032x}" for c in columns])
    assert field.ok and field.data["irreducible"]
    field_id = field.data["field_id"]
    closure = kernel.gf2m_closure_roots(
        field_id, [[f"{r:032x}" for r in mask_rows(m)] for m, _ in strong])
    assert closure.ok and closure.data["all_closures_valid"]

    # validation on fresh states: sign and magnitude must reproduce
    vrng = random.Random(0x5EED_11)
    confirmed = 0
    for (mask, train_rho), candidate in zip(strong, closure.data["candidates"]):
        assert len(candidate["roots"]) == 1
        root = candidate["roots"][0]
        rows = [int(r, 16)
                for r in kernel.gf2m_root_jump_rows(field_id, root).data["rows"]]
        score = 0
        for _ in range(n_validate):
            s0 = vrng.getrandbits(128)
            s1 = TransitionField.apply_rows(rows, s0)
            x = 0
            for i, (word, bit) in enumerate(POSITIONS):
                x |= ((s0 >> (bit + 64 * word)) & 1) << i
                x |= ((s1 >> (bit + 64 * word)) & 1) << (i + 9)
            out0 = (rotl(((s0 & MASK64) * 5) & MASK64, 7) * 9) & MASK64
            out1 = (rotl(((s1 & MASK64) * 5) & MASK64, 7) * 9) & MASK64
            target = ((out0 - out1) & MASK64) >> 63
            score += 1 if ((mask & x).bit_count() & 1) == target else -1
        fresh_rho = score / n_validate
        if abs(fresh_rho) > 4 / n_validate ** 0.5:
            assert (fresh_rho < 0) == (train_rho < 0)
            confirmed += 1
        # mutation probe: one-bit-mutated root must fail the closure
        coeff = candidate["coefficients"]
        mutated = f"{int(root, 16) ^ 1:032x}"
        product = kernel.gf2m_compute(field_id, "mul", [coeff[1], mutated]).data["result"]
        residual = kernel.gf2m_compute(field_id, "add", [coeff[0], product]).data["result"]
        assert int(residual, 16) != 0
    assert confirmed >= 3

    # alpha action: root 2 induces exactly one generator step
    rows2 = [int(r, 16) for r in kernel.gf2m_root_jump_rows(field_id, "2").data["rows"]]
    probe = vrng.getrandbits(128)
    assert TransitionField.apply_rows(rows2, probe) == xoro_step(probe)


# ---------------------------------------------------------------------------
# Full xoroshiro128** closure reproduction against the shipped reference
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REFERENCE.is_file(), reason="reference closure_roots.json not present")
def test_xoroshiro128starstar_closure_reproduction():
    kernel = MathKernel()
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))

    columns = [xoro_step(1 << i) for i in range(128)]
    field = kernel.gf2m_from_transition([f"{c:032x}" for c in columns])
    assert field.ok
    assert field.data["reduction"] == reference["RED"]
    field_id = field.data["field_id"]

    # alpha action: root 2 induces exactly one generator step
    jump = kernel.gf2m_root_jump_rows(field_id, "2")
    assert [int(r, 16) for r in jump.data["rows"]] == rows_from_cols(columns)

    # all 200 reference roots reproduced from their masks, residuals zero
    row_sets = [[f"{r:032x}" for r in mask_rows(e["mask"])] for e in reference["roots"]]
    closure = kernel.gf2m_closure_roots(field_id, row_sets)
    assert closure.ok and closure.data["all_closures_valid"]
    for entry, candidate in zip(reference["roots"], closure.data["candidates"]):
        rows = mask_rows(entry["mask"])
        assert f"{rows[0]:032x}" == entry["Arow"]
        assert f"{rows[1]:032x}" == entry["Brow"]
        assert candidate["roots"] == [entry["root"]]
        assert candidate["residuals"] == ["0" * 32]

    # mutation probe: a one-bit-mutated root must fail the closure
    first = reference["roots"][0]
    coeff = kernel.gf2m_closure_roots(
        field_id, [[f"{r:032x}" for r in mask_rows(first["mask"])]]
    ).data["candidates"][0]["coefficients"]
    mutated = int(first["root"], 16) ^ 1
    product = kernel.gf2m_compute(field_id, "mul", [coeff[1], f"{mutated:032x}"]).data["result"]
    residual = kernel.gf2m_compute(field_id, "add", [coeff[0], product]).data["result"]
    assert int(residual, 16) != 0
