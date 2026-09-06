# Sub-skill: Finite dynamics and PRNG modules

Part of the `mathkernel` skill. Load when working directly with the
finite-dynamics modules: `gf2m`, `transforms`, `cumulants`,
`finite_fourier`, `koopman`, `relations`, `stochastic_koopman`, and
`phylogenetic_tensor`.

## gf2m — binary fields GF(2^m)

```python
from mathkernel.gf2m import GF2mField, gf2m_from_transition
F = GF2mField(8, 0x1B)                   # degree, reduction part of x^8+x^4+x^3+x+1
a = F.mul(0x53, 0xCA); F.inv(a); F.sqrt(a); F.trace(a)
F.quadratic_roots(u, v)                   # x^2 + u x + v = 0
# from a GF(2)-linear transition (dual-orbit cyclic basis + minpoly):
F2 = gf2m_from_transition(columns)        # columns: bit-packed ints
```

Degrees up to 1024 run on njit n-limb kernels (`gf2m_fast`); larger fall
back to pure Python. Irreducibility is Rabin-tested.

GF(2) matrix helpers (bit-packed rows): `gf2_rank`, `gf2_nullspace`,
`gf2_apply`, `gf2_apply_transpose`, `gf2_rows_mul`, `gf2_rows_power`,
`carryfree_cols`, `berlekamp_massey`.

## transforms — FWHT

```python
from mathkernel.transforms import fwht
fwht([1, -1, 1, -1])      # exact unnormalized; int64 njit fast path,
                          # arbitrary-precision fallback beyond the bound
```

## cumulants — partition lattice

```python
from mathkernel.cumulants import (cumulant_from_moments,
    moment_from_cumulants, block_moments_from_samples,
    joint_cumulant_from_samples, set_partitions)
# moments keyed by subset bitmask over coordinates 0..d-1
k = cumulant_from_moments({0b01: m0, 0b10: m1, 0b11: m01}, d=2)
```

Generic over any value type supporting `+`/`*` (Fraction, complex,
CyclotomicNumber, numpy scalars). Float/complex sample columns take a
vectorized numpy path; int/Fraction stay exact.

## finite_fourier — exact cyclotomic arithmetic

```python
from mathkernel.finite_fourier import (CyclotomicNumber, dft_zm,
    transfer_transform, two_point_difference, measure_fourier,
    orbit_correction)
z = CyclotomicNumber.zeta(8)              # exact element of Q(zeta_8)
coeffs = dft_zm([1, 2, 3, 4])             # exact DFT over Z_4
```

`CyclotomicNumber` supports `+ - * /`, `conjugate`, `norm_sq`,
`to_rational`, `to_complex`. Numeric complex input to `dft_zm` uses the
numpy FFT instead.

## koopman — finite partially observed systems

```python
from mathkernel.koopman import (FiniteSystem, walsh_basis, cyclic_basis,
    koopman_matrix, transfer_matrix, mode_visibility, lagged_tensor_value,
    observed_statistic, transport_diagnostics)
sys = FiniteSystem.uniform(8, transition, observation)
basis, norm_sq = walsh_basis(3)           # or cyclic_basis(m)
Q = koopman_matrix(sys, basis, norm_sq)   # exact
rho = mode_visibility(sys, basis, norm_sq)
```

Each exact function has a `*_numeric` twin (complex128, CuPy GPU when
usable): `koopman_matrix_numeric`, `transfer_matrix_numeric`,
`mode_visibility_numeric`, `lagged_tensor_numeric`,
`observed_statistic_numeric`, `transport_diagnostics_numeric`. Use numeric
for large systems, then confirm decisive zeros with the exact path.

Higher-level: `static_cumulant_tensor`, `path_tensor_value` (Koopman path
representation), `contraction_value` (observed statistic via basis
contraction).

## stochastic_koopman — arbitrary finite laws and Markov paths

```python
from fractions import Fraction
from mathkernel import FiniteJointLaw, FiniteMarkovKernel

law = FiniteJointLaw.from_dense((2, 2), [
    Fraction(1, 3), Fraction(1, 6),
    Fraction(1, 6), Fraction(1, 3),
])
# Universal observation-transfer contractions do not require deterministic
# dynamics; a finite joint law is sufficient.

kernel = FiniteMarkovKernel.from_rows([
    [Fraction(3, 4), Fraction(1, 4)],
    [Fraction(1, 3), Fraction(2, 3)],
])
m = kernel.path_moment(
    (Fraction(1, 2), Fraction(1, 2)),
    times=(1, 2),
    functions=((-1, 1), (-1, 1)),
)
```

Markov path moments use ordered propagation with multiplication operators;
they are not obtained by independently replacing deterministic Koopman powers
with powers of a transition matrix. `RationalMarkovDilation` realizes a
rational kernel as a deterministic map on state plus a finite noise stream.

## phylogenetic_tensor — branching General Markov laws

```python
from mathkernel import (
    FiniteMarkovTree, FiniteObservationChannel,
    edge_flattening_certificate, observation_flattening_certificate,
)

# Build with FiniteMarkovTree.from_edges(...), then obtain an exact leaf law.
law = tree.leaf_joint_law()
edge = edge_flattening_certificate(tree, internal_child)
assert edge.three_factorization_exact
assert edge.flattening_rank <= edge.transition_rank

channel = FiniteObservationChannel.from_rows(rows)
obs = observation_flattening_certificate(law, (channel,) * 4, (0, 1))
assert obs.identity_exact       # F_Y = A_left F_X A_right^T
```

`FiniteMarkovTree` supports heterogeneous vertex alphabets and exact rational
sum-product/message passing. `FiniteObservationChannel` supports stochastic or
deterministic observations, exact pullback, exact left inverses when full
latent rank, rank-deficiency collision witnesses, and conditionally independent
sensor fusion. `tensor_flattening`, `recover_latent_joint_law`,
`minimum_collectively_injective_channel_sets`, and
`singular_value_observation_bounds` provide the associated exact and numerical
diagnostics. These APIs are currently library-only; they are not registered as
MCP tools.

## relations — closure search

```python
from mathkernel.relations import (cyclic_closure, cyclic_closure_order,
    binary_closure, binary_closure_order, is_irreducible_cyclic,
    is_irreducible_binary)
cyclic_closure([1, 5], 97, 10)            # tuples k: sum k_j a_j == 0 (mod m)
binary_closure(rows, width=4, d=2, k_step=1, max_weight=2)
```

njit meet-in-the-middle kernels handle the int64/uint64 domain (cyclic:
`d*H*m < 2^62`; binary: `width <= 63`); larger instances fall back to
arbitrary-precision Python. Enumeration is budget-capped
(`MAX_ENUMERATION`); a `ValueError` means lower the weight/order.

## Facade equivalents

The deterministic finite-dynamics, Fourier, closure, GF(2), and cumulant APIs
are also reachable through `MathKernel` methods (`cumulant_compute`,
`finite_system_create`, `koopman_*`, `finite_fourier_compute`,
`closure_search`, `gf2m_*`, `gf2_*`, `fwht`), which add validation, derivation
tracking, and trust labels. `stochastic_koopman` and `phylogenetic_tensor` are
currently direct library APIs only.
