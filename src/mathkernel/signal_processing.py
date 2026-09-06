# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Finite sampled signals and bounded continuous-signal sampling."""
from __future__ import annotations
from functools import lru_cache
from typing import Literal
import sympy as sp
from pydantic import model_validator
from .engineering import (EngineeringModel, arithmetic_trust, cap_trust,
    checked_result, numeric_array, sympy_samples, validate_scalars, exact_zero)
from .units import parse_unit


class DiscreteSignal(EngineeringModel):
    samples: tuple[sp.Expr, ...]
    sample_rate: sp.Expr = sp.Integer(1)  # Hz
    start: sp.Expr = sp.Integer(0)  # seconds
    unit: str = ""  # dimensionless by default
    input_trust: str = "exact"
    transformations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_signal(self):
        if not self.samples:
            raise ValueError("signal requires at least one sample")
        validate_scalars((*self.samples, self.sample_rate, self.start))
        if self.sample_rate.is_positive is not True or self.start.is_real is not True:
            raise ValueError("sample_rate must be positive real Hz and start real seconds")
        parse_unit(self.unit)
        cap_trust(self.input_trust)
        return self


class ContinuousSignal(EngineeringModel):
    """A scalar analytic signal on a finite interval.

    The expression is data, never an executable Python callback.  Sampling does
    not imply bandlimiting or a reconstruction guarantee.
    """
    expression: sp.Expr
    variable: str = "t"
    start: sp.Expr = sp.Integer(0)
    end: sp.Expr
    unit: str = ""
    time_unit: Literal["s"] = "s"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_continuous(self):
        if not self.variable.isidentifier() or self.variable.startswith("_"):
            raise ValueError("variable must be a public identifier")
        validate_scalars((self.expression, self.start, self.end))
        if any(symbol.name != self.variable for symbol in self.expression.free_symbols):
            raise ValueError("continuous signal expression may use only its declared variable")
        if self.start.free_symbols or self.end.free_symbols or self.start.is_real is not True or self.end.is_real is not True:
            raise ValueError("continuous signal bounds must be concrete real values")
        if (self.end-self.start).is_positive is not True:
            raise ValueError("continuous signal end must be greater than start")
        parse_unit(self.unit)
        cap_trust(self.input_trust)
        return self


def sample_continuous(signal, *, sample_rate, count, mode="exact", max_output=1_000_000):
    validate_scalars((sample_rate,), real=True)
    if sample_rate.free_symbols or sample_rate.is_positive is not True:
        raise ValueError("sample_rate must be a concrete positive value in Hz")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= max_output:
        raise ValueError("count must be a positive integer within the output limit")
    times = tuple(signal.start + sp.Rational(i)/sample_rate for i in range(count))
    if times[-1] > signal.end:
        raise ValueError("requested uniform samples extend beyond the signal interval")
    variable = next(iter(signal.expression.free_symbols), sp.Symbol(signal.variable))
    trust = arithmetic_trust((signal.expression, signal.start, signal.end, sample_rate), signal.input_trust)
    if mode == "exact":
        values = tuple(sp.simplify(signal.expression.subs(variable, t)) for t in times)
        checks = {"substitution_identity": all(sp.simplify(v-signal.expression.subs(variable, t)) == 0 for v, t in zip(values, times))}
        method, residual = "exact_uniform_substitution", None
    elif mode == "numeric":
        import numpy as np
        fn = sp.lambdify(variable, signal.expression, modules="numpy")
        raw = np.asarray(fn(np.asarray([float(t) for t in times])), dtype=complex)
        if raw.ndim == 0:
            raw = np.full(count, raw, dtype=complex)
        if raw.shape != (count,) or not np.all(np.isfinite(raw)):
            raise ValueError("continuous expression did not produce finite scalar samples")
        reference = np.asarray([complex(signal.expression.subs(variable, t).evalf(17)) for t in times])
        residual = float(np.max(np.abs(raw-reference)))
        tolerance = float(64*np.finfo(float).eps*max(1, np.max(np.abs(reference))))
        values = sympy_samples(raw)
        checks = {"scalar_substitution_consistency": bool(residual <= tolerance)}
        trust, method = cap_trust(trust, "numeric"), "numpy_vectorized_uniform_sampling"
    else:
        raise ValueError("mode must be exact or numeric")
    derived = DiscreteSignal(samples=values, sample_rate=sample_rate, start=signal.start,
        unit=signal.unit, input_trust=trust, transformations=("uniform sampling of ContinuousSignal",))
    result = checked_result("sample", values, method=method, trust=trust, checks=checks,
        residual=residual, precision=53 if mode == "numeric" else None,
        details={"sample_rate": sample_rate, "count": count, "start": signal.start,
                 "last_sample_time": times[-1], "interval_end": signal.end,
                 "time_unit": signal.time_unit, "endpoint_policy": "include start; fixed count",
                 "bandlimited_assumption": False, "reconstruction_guarantee": False})
    return result, derived


class Spectrum(EngineeringModel):
    bins: tuple[sp.Expr, ...]
    sample_rate: sp.Expr
    start: sp.Expr = sp.Integer(0)
    unit: str = ""
    normalization: Literal["backward", "forward", "ortho"] = "backward"
    input_trust: str = "exact"
    convention: Literal["exp(-2*pi*i*k*n/N)"] = "exp(-2*pi*i*k*n/N)"

    @model_validator(mode="after")
    def validate_spectrum(self):
        if not self.bins:
            raise ValueError("spectrum requires at least one bin")
        validate_scalars((*self.bins, self.sample_rate, self.start))
        if self.sample_rate.is_positive is not True or self.start.is_real is not True:
            raise ValueError("invalid sample rate or time origin")
        parse_unit(self.unit)
        cap_trust(self.input_trust)
        return self


class Filter(EngineeringModel):
    # H(z) = (b[0] + b[1]/z + ...)/(a[0] + a[1]/z + ...).
    numerator: tuple[sp.Expr, ...]
    denominator: tuple[sp.Expr, ...] = (sp.Integer(1),)
    sample_rate: sp.Expr = sp.Integer(1)
    input_trust: str = "exact"
    sos: tuple[tuple[sp.Expr, ...], ...] = ()

    @model_validator(mode="after")
    def validate_filter(self):
        if not self.numerator or not self.denominator:
            raise ValueError("filter requires numerator and denominator coefficients")
        validate_scalars((*self.numerator, *self.denominator, self.sample_rate))
        if self.denominator[0].is_zero is not False:
            raise ValueError("filter a[0] must be provably nonzero")
        if self.sample_rate.is_positive is not True:
            raise ValueError("sample_rate must be positive Hz")
        if self.sos:
            if any(len(row) != 6 or (row[3]-1).is_zero is not True for row in self.sos):
                raise ValueError("SOS rows require six coefficients with a0=1")
            validate_scalars(x for row in self.sos for x in row)
        cap_trust(self.input_trust)
        return self


def _scale(n, norm, inverse):
    if norm not in {"backward", "forward", "ortho"}:
        raise ValueError("normalization must be backward, forward, or ortho")
    if norm == "ortho":
        return 1/sp.sqrt(n)
    return sp.Rational(1, n) if (norm == "backward") == inverse else sp.Integer(1)


@lru_cache(maxsize=16)
def _roots(n, inverse):
    # One immutable table per length, not N^2 symbolic exponentials.
    sign = 1 if inverse else -1
    return tuple(sp.expand_complex(sp.exp(sign*2*sp.pi*sp.I*k/n)) for k in range(n))


def exact_dft(samples, *, inverse=False, normalization="backward"):
    n = len(samples)
    roots, scale = _roots(n, inverse), _scale(n, normalization, inverse)
    if n > 1 and n & (n-1) == 0 and all(x.is_Rational for x in samples):
        # Exact cyclotomic reduction zeta^(N/2)=-1 aggregates rational
        # coefficients before constructing radicals. Avoids costly per-bin
        # general simplification, which dominates small symbolic FFTs.
        half = n//2
        output = []
        active = [(j, x) for j, x in enumerate(samples) if x != 0]
        for k in range(n):
            coefficients = [sp.S.Zero]*half
            for j, x in active:
                power = j*k % n
                coefficients[power % half] += x if power < half else -x
            output.append(sp.expand(scale*sum((c*roots[j] for j, c in enumerate(coefficients) if c != 0), sp.S.Zero)))
        return tuple(output)
    if n > 16 and n & (n-1) == 0:
        # Radix-2 butterflies keep exact ring arithmetic while reducing work
        # from quadratic sums to N log2(N); no float conversion is involved.
        bits = (n-1).bit_length()
        out = [samples[int(f"{i:0{bits}b}"[::-1], 2)] for i in range(n)] if n > 1 else list(samples)
        width = 2
        while width <= n:
            half = width//2
            for base in range(0, n, width):
                for j in range(half):
                    even = out[base+j]
                    odd = sp.expand(roots[j*n//width]*out[base+j+half])
                    out[base+j], out[base+j+half] = sp.expand(even+odd), sp.expand(even-odd)
            width *= 2
        return tuple(sp.simplify(scale*x) for x in out)
    active = [(j, x) for j, x in enumerate(samples) if x != 0]
    return tuple(sp.simplify(scale*sum((x*roots[(j*k) % n] for j, x in active), sp.S.Zero))
                 for k in range(n))



def dft(signal, *, mode="exact", normalization="backward", inverse=False,
        precision=53, max_exact=64, max_high_precision=256):
    samples = signal.bins if inverse else signal.samples
    if inverse:
        normalization = signal.normalization
    n = len(samples)
    scale = _scale(n, normalization, inverse)
    verification_tolerance = None
    input_trust = arithmetic_trust((*samples, signal.sample_rate, signal.start), signal.input_trust)
    if mode == "exact":
        if n > max_exact:
            raise ValueError(f"exact DFT exceeds max_exact_dft_size={max_exact}; choose mode='numeric' explicitly")
        values = exact_dft(samples, inverse=inverse, normalization=normalization)
        restored = exact_dft(values, inverse=not inverse, normalization=normalization)
        verdicts = [exact_zero(x-y) for x, y in zip(samples, restored)]
        verdict = False if False in verdicts else True if all(v is True for v in verdicts) else None
        checks = {"round_trip": verdict}
        trust, method, residual = input_trust, "exact_dft_roots_of_unity", None
    elif mode == "numeric":
        import numpy as np
        if precision != 53:
            if not 54 <= precision <= 4096 or n > max_high_precision:
                raise ValueError("high-precision DFT requires 54..4096 bits and a bounded transform length")
            import mpmath as mp
            ctx = mp.mp.clone()
            ctx.prec = precision
            digits = int(precision*0.30103)+8
            data = [ctx.mpc(str(sp.re(x).evalf(digits)), str(sp.im(x).evalf(digits))) for x in samples]
            root = ctx.exp((1 if inverse else -1)*2*ctx.pi*ctx.j/n)
            norm_scale = ctx.mpf(str(scale.evalf(digits)))
            raw = [norm_scale*ctx.fsum(data[j]*root**((j*k) % n) for j in range(n)) for k in range(n)]
            values = tuple(sp.Float(str(x.real), digits) + sp.I*sp.Float(str(x.imag), digits) for x in raw)
            # Parseval is independently evaluated; it is a residual, not an error bound.
            energy = ctx.fsum(abs(x)**2 for x in data)
            residual = abs(ctx.fsum(abs(x)**2 for x in raw)-n*norm_scale**2*energy)
            checks = {"parseval_consistency": bool(residual <= ctx.ldexp(max(1, energy), -precision+12))}
            verification_tolerance = str(ctx.ldexp(max(1, energy), -precision+12))
            residual = str(residual)
            trust, method = cap_trust(input_trust, "numeric_high_precision"), "mpmath_dft"
        else:
            data = numeric_array(samples)
            function = np.fft.ifft if inverse else np.fft.fft
            raw = function(data, norm=normalization)
            values = sympy_samples(raw)
            restored = (np.fft.fft if inverse else np.fft.ifft)(raw, norm=normalization)
            residual = float(np.max(np.abs(restored-data)))
            budget = 64*np.finfo(float).eps*max(1, np.log2(n))*max(1, float(np.max(np.abs(data))))
            verification_tolerance = float(budget)
            checks = {"round_trip_numeric": bool(residual <= budget)}
            trust, method = cap_trust(input_trust, "numeric"), "numpy_pocketfft"
    else:
        raise ValueError("mode must be exact or numeric")
    metadata = dict(sample_rate=signal.sample_rate, size=n, overlap=0, padding=0,
                    window="boxcar", normalization=normalization,
                    convention="exp(-2*pi*i*k*n/N)", frequency_unit="Hz",
                    frequency_order="unshifted", time_origin=signal.start,
                    inverse=inverse, precision_bits=precision if mode == "numeric" else None,
                    source_transformations=list(getattr(signal, "transformations", ())),
                    verification_absolute_tolerance=verification_tolerance)
    if inverse:
        derived = DiscreteSignal(samples=values, sample_rate=signal.sample_rate,
                                 start=signal.start, unit=signal.unit, input_trust=trust)
    else:
        derived = Spectrum(bins=values, sample_rate=signal.sample_rate, start=signal.start,
                           unit=signal.unit, normalization=normalization, input_trust=trust)
    result = checked_result("idft" if inverse else "dft", values, method=method, trust=trust,
                            checks=checks, details=metadata, residual=residual,
                            precision=precision if mode == "numeric" else None)
    return result, derived


def convolve(left, right, *, mode="exact", correlation=False, max_work=1_000_000):
    if sp.simplify(left.sample_rate-right.sample_rate) != 0:
        raise ValueError("sample rates differ; resample explicitly before composition")
    a, b = left.samples, right.samples
    start = left.start + right.start
    if correlation:
        b = tuple(sp.conjugate(x) for x in reversed(b))
        start = left.start-right.start-sp.Rational(len(b)-1)/left.sample_rate
    if len(a)+len(b)-1 > max_work:
        raise ValueError("convolution output exceeds resource limit")
    trust = cap_trust(arithmetic_trust((*a, left.sample_rate), left.input_trust),
                      arithmetic_trust((*b, right.sample_rate), right.input_trust))
    if mode == "exact":
        active_a = [(i, x) for i, x in enumerate(a) if x != 0]
        active_b = [(i, x) for i, x in enumerate(b) if x != 0]
        if len(active_a)*len(active_b) > max_work:
            raise ValueError("exact convolution work budget exceeded; choose numeric mode explicitly")
        output = [sp.S.Zero]*(len(a)+len(b)-1)
        for i, x in active_a:
            for j, y in active_b:
                output[i+j] += x*y
        values = tuple(sp.expand(x) for x in output)
        # Polynomial evaluations are a diagnostic, not a completeness certificate.
        checks = {"dc_sum_identity": sp.simplify(sum(values)-sum(a)*sum(b)) == 0}
        method, residual = "sparse_exact_convolution", None
    elif mode == "numeric":
        from scipy import signal as ss
        import numpy as np
        x, y = numeric_array(a), numeric_array(b)
        # Overlap-add avoids a single huge FFT for strongly unequal lengths.
        if max(len(x), len(y)) >= 16*min(len(x), len(y)) and min(len(x), len(y)) > 32:
            raw, selected = ss.oaconvolve(x, y), "overlap_add"
        else:
            selected = ss.choose_conv_method(x, y, mode="full")
            raw = ss.convolve(x, y, method=selected)
        values = sympy_samples(raw)
        residual = float(abs(np.sum(raw)-np.sum(x)*np.sum(y)))
        bound = 128*np.finfo(float).eps*max(1, float(np.sum(abs(x))*np.sum(abs(y))))
        checks = {"dc_sum_consistency": bool(residual <= bound)}
        trust, method = cap_trust(trust, "numeric"), f"scipy_convolution_{selected}"
    else:
        raise ValueError("mode must be exact or numeric")
    # Units are parsed and combined without ambiguous slash concatenation.
    combined = parse_unit(left.unit).dimension*parse_unit(right.unit).dimension
    # Preserve magnitude units as a product; parse_unit's grammar has no parentheses.
    unit = _product_unit(left.unit, right.unit)
    derived = DiscreteSignal(samples=values, sample_rate=left.sample_rate,
                             start=start, unit=unit, input_trust=trust)
    operation = "correlation" if correlation else "convolution"
    return checked_result(operation, values, method=method, trust=trust, checks=checks,
                          residual=residual, precision=53 if mode == "numeric" else None,
                          details={"mode": "full", "padding": "zero", "normalization": "none",
                                   "sample_rate": left.sample_rate, "start": start,
                                   "unit": unit, "dimension": str(combined),
                                   "verification_absolute_tolerance": float(bound) if mode == "numeric" else None}), derived


def _product_unit(left, right):
    # Canonical token powers preserve prefixed-unit scaling (e.g. mV if registered).
    powers = {}
    for text in (left, right):
        for i, segment in enumerate(text.split("/")):
            for token in segment.split("*"):
                if token.strip():
                    name, _, exponent = token.strip().partition("^")
                    powers[name] = powers.get(name, 0)+(int(exponent or 1)*(1 if i == 0 else -1))
    return "*".join(f"{name}^{power}" for name, power in sorted(powers.items()) if power)


def window(signal, *, kind="hann", periodic=True, mode="exact"):
    n = len(signal.samples)
    if kind not in {"hann", "hamming", "boxcar"}:
        raise ValueError("window must be hann, hamming, or boxcar")
    if mode == "numeric":
        from scipy.signal import get_window
        weights = get_window(kind, n, fftbins=periodic)
        values = sympy_samples(numeric_array(signal.samples)*weights)
        trust = cap_trust(signal.input_trust, "numeric")
        derived = signal.model_copy(update={"samples": values, "input_trust": trust,
            "transformations": (*signal.transformations, f"{kind} window; periodic={periodic}")})
        return checked_result("window", values, method="vectorized_window_product", trust=trust,
            precision=53, details={"window": kind, "periodic": periodic, "size": n,
                                   "coherent_gain": float(weights.sum()/n)}), derived
    if mode != "exact":
        raise ValueError("mode must be exact or numeric")
    denominator = n if periodic else max(n-1, 1)
    weights = tuple(sp.S.One if kind == "boxcar" or n == 1 else
                    (1-sp.cos(2*sp.pi*j/denominator))/2 if kind == "hann" else
                    sp.Rational(27, 50)-sp.Rational(23, 50)*sp.cos(2*sp.pi*j/denominator)
                    for j in range(n))
    values = tuple(x*w for x, w in zip(signal.samples, weights))
    trust = arithmetic_trust((*values, signal.sample_rate), signal.input_trust)
    derived = signal.model_copy(update={"samples": values, "input_trust": trust,
        "transformations": (*signal.transformations, f"{kind} window; periodic={periodic}")})
    return checked_result("window", values, method="window_pointwise_product", trust=trust,
                          details={"window": kind, "periodic": periodic, "size": n,
                                   "coherent_gain": sum(weights)/n}), derived


def apply_filter(filt, signal, *, mode="exact", max_work=1_000_000):
    if sp.simplify(filt.sample_rate-signal.sample_rate) != 0:
        raise ValueError("filter and signal sample rates differ")
    b, a, x = filt.numerator, filt.denominator, signal.samples
    trust = cap_trust(arithmetic_trust((*a, *b, filt.sample_rate), filt.input_trust),
                      arithmetic_trust((*x, signal.sample_rate), signal.input_trust))
    if len(x)*(len(a)+len(b)) > max_work:
        raise ValueError("filter recurrence verification exceeds work budget")
    if mode == "exact":
        y = []
        for n in range(len(x)):
            rhs = sum((b[j]*x[n-j] for j in range(min(n+1, len(b)))), sp.S.Zero)
            past = sum((a[j]*y[n-j] for j in range(1, min(n+1, len(a)))), sp.S.Zero)
            y.append(sp.cancel((rhs-past)/a[0]))
        residuals = [sp.simplify(sum(a[j]*y[n-j] for j in range(min(n+1, len(a))))-
                                sum(b[j]*x[n-j] for j in range(min(n+1, len(b))))) for n in range(len(x))]
        checks = {"difference_equation": all(r == 0 for r in residuals)}
        values, method, residual = tuple(y), "exact_causal_recurrence", None
    elif mode == "numeric":
        import numpy as np
        from scipy.signal import lfilter, sosfilt
        xn, an, bn = numeric_array(x), numeric_array(a), numeric_array(b)
        if filt.sos:
            sos = numeric_array(tuple(v for row in filt.sos for v in row)).reshape((-1, 6))
            yn = sosfilt(sos, xn)
        else:
            yn = lfilter(bn, an, xn)
        lhs = np.convolve(an, yn)[:len(xn)]
        rhs = np.convolve(bn, xn)[:len(xn)]
        residual = float(np.max(abs(lhs-rhs)))
        budget = 128*np.finfo(float).eps*max(1, float(np.max(abs(lhs))), float(np.max(abs(rhs))))
        values = sympy_samples(yn)
        checks = {"difference_equation_numeric": bool(residual <= budget)}
        trust, method = cap_trust(trust, "numeric"), "scipy_sosfilt" if filt.sos else "scipy_lfilter"
    else:
        raise ValueError("mode must be exact or numeric")
    derived = signal.model_copy(update={"samples": values, "input_trust": trust})
    return checked_result("filter", values, method=method, trust=trust, checks=checks,
                          residual=residual, precision=53 if mode == "numeric" else None,
                          details={"initial_state": "zero", "sample_rate": signal.sample_rate,
                                   "difference_equation": "sum(a[j]*y[n-j]) = sum(b[j]*x[n-j])",
                                   "verification_absolute_tolerance": float(budget) if mode == "numeric" else None}), derived


def stft(signal, *, size=256, overlap=128, window_kind="hann", max_output=1_000_000):
    import numpy as np
    from scipy.signal import get_window
    if not 1 <= size <= len(signal.samples) or not 0 <= overlap < size:
        raise ValueError("STFT requires 1 <= size <= sample count and 0 <= overlap < size")
    frames = 1+(len(signal.samples)-size)//(size-overlap)
    if frames*size > max_output:
        raise ValueError("STFT output exceeds resource limit")
    if window_kind not in {"hann", "hamming", "boxcar"}:
        raise ValueError("unsupported STFT window")
    x = numeric_array(signal.samples)
    windows = np.lib.stride_tricks.sliding_window_view(x, size)[::size-overlap]
    weights = get_window(window_kind, size, fftbins=True)
    transformed = np.fft.fft(windows*weights, axis=1)
    restored = np.fft.ifft(transformed, axis=1)
    residual = float(np.max(abs(restored-windows*weights)))
    trust = cap_trust(signal.input_trust, "numeric")
    values = tuple(sympy_samples(row) for row in transformed)
    return checked_result("stft", values, method="batched_pocketfft_strided_frames", trust=trust,
        checks={"frame_round_trip": bool(np.allclose(restored, windows*weights, rtol=1e-12, atol=1e-12))},
        residual=residual, precision=53,
        details={"sample_rate": signal.sample_rate, "size": size, "overlap": overlap,
                 "hop": size-overlap, "window": window_kind, "periodic_window": True,
                 "padding": 0, "boundary": "drop_incomplete_frame", "normalization": "backward",
                 "frame_starts": [signal.start+sp.Rational(i*(size-overlap))/signal.sample_rate for i in range(frames)],
                 "frequency_order": "unshifted", "convention": "exp(-2*pi*i*k*n/N)",
                 "verification_rtol": 1e-12, "verification_atol": 1e-12})


def resample(signal, *, up, down, max_output=1_000_000):
    import math
    from scipy.signal import resample_poly
    if up < 1 or down < 1:
        raise ValueError("resampling factors must be positive")
    divisor = math.gcd(up, down)
    up, down = up//divisor, down//divisor
    if max(up, down)*20+1 > max_output or (len(signal.samples)*up+down-1)//down > max_output:
        raise ValueError("resampling filter/output exceeds resource limit")
    raw = resample_poly(numeric_array(signal.samples), up, down, window=("kaiser", 5.0), padtype="constant")
    values = sympy_samples(raw)
    trust = cap_trust(signal.input_trust, "numeric")
    derived = signal.model_copy(update={"samples": values, "sample_rate": signal.sample_rate*sp.Rational(up, down),
                                        "input_trust": trust})
    result = checked_result("resample", values, method="scipy_polyphase_fir", trust=trust,
        details={"up": up, "down": down, "sample_rate": derived.sample_rate,
                 "window": "kaiser(beta=5)", "padding": "constant_zero", "normalization": "amplitude",
                 "aliasing_bound": None, "method": "polyphase_fir"}, precision=53)
    result.diagnostics.append("No bandlimit assumption or rigorous aliasing/error bound is established.")
    return result, derived


class FilterDesign(EngineeringModel):
    family: Literal["butterworth", "chebyshev1", "chebyshev2", "bessel", "elliptic", "fir_window"] = "butterworth"
    order: int = 4
    cutoff: tuple[sp.Expr, ...]
    sample_rate: sp.Expr
    kind: Literal["lowpass", "highpass", "bandpass", "bandstop"] = "lowpass"
    passband_ripple: sp.Expr = sp.Integer(1)  # dB
    stopband_attenuation: sp.Expr = sp.Integer(40)  # dB
    window: Literal["hann", "hamming", "blackman", "boxcar"] = "hamming"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_design(self):
        from .engineering import positive_integer
        positive_integer(self.order, "filter order", 8)
        validate_scalars((*self.cutoff, self.sample_rate, self.passband_ripple, self.stopband_attenuation), real=True)
        expected = 2 if self.kind in {"bandpass", "bandstop"} else 1
        if len(self.cutoff) != expected or self.sample_rate.is_positive is not True:
            raise ValueError("cutoff count/sample rate does not match filter design")
        if any(x.is_positive is not True or (self.sample_rate/2-x).is_positive is not True for x in self.cutoff):
            raise ValueError("cutoffs must lie strictly between zero and Nyquist in Hz")
        if expected == 2 and (self.cutoff[1]-self.cutoff[0]).is_positive is not True:
            raise ValueError("band cutoffs must be increasing")
        if self.passband_ripple.is_positive is not True or self.stopband_attenuation.is_positive is not True:
            raise ValueError("ripple/attenuation must be positive dB")
        if any(x.free_symbols for x in (*self.cutoff, self.sample_rate, self.passband_ripple, self.stopband_attenuation)):
            raise ValueError("filter design requires concrete specifications")
        cap_trust(self.input_trust)
        return self


def design_filter(spec):
    import numpy as np
    import scipy
    from scipy import signal as ss
    cutoff_values = numeric_array(spec.cutoff, real=True)
    # SciPy 1.18 requires a scalar Wn for one-edge designs; older releases
    # tolerated a length-one array. Preserve arrays only for band designs.
    cutoff = float(cutoff_values[0]) if len(cutoff_values) == 1 else cutoff_values
    if spec.family == "fir_window":
        pass_zero = {"lowpass": True, "highpass": False, "bandpass": False, "bandstop": True}[spec.kind]
        b = ss.firwin(spec.order+1, cutoff, window=spec.window, pass_zero=pass_zero, fs=float(spec.sample_rate))
        a, sos, p = np.asarray([1.]), np.empty((0, 6)), np.asarray([])
        implementation = "linear_phase_windowed_fir"
    else:
        family = {"butterworth": "butter", "chebyshev1": "cheby1", "chebyshev2": "cheby2",
                  "bessel": "bessel", "elliptic": "ellip"}[spec.family]
        z, p, gain = ss.iirfilter(spec.order, cutoff, rp=float(spec.passband_ripple),
            rs=float(spec.stopband_attenuation), btype=spec.kind, ftype=family,
            fs=float(spec.sample_rate), output="zpk")
        sos = ss.zpk2sos(z, p, gain)
        b, a = ss.zpk2tf(z, p, gain)
        implementation = "second_order_sections"
    trust = cap_trust(spec.input_trust, "numeric")
    derived = Filter(numerator=sympy_samples(b), denominator=sympy_samples(a),
        sample_rate=spec.sample_rate, input_trust=trust,
        sos=tuple(sympy_samples(row) for row in sos))
    omega = np.linspace(0, np.pi, 257)
    _, direct = ss.freqz(b, a, worN=omega)
    if len(sos):
        _, sections = ss.sosfreqz(sos, worN=omega)
        residual = float(np.max(abs(direct-sections)))
        tolerance = float(1024*np.finfo(float).eps*max(1, np.max(abs(direct)))*max(1, len(sos)))
        checks = {"sos_direct_frequency_consistency": bool(residual <= tolerance)}
    else:
        # Linear phase is independently checked from coefficient symmetry.
        residual = float(np.max(abs(b-b[::-1])))
        tolerance = float(64*np.finfo(float).eps*max(1, np.max(abs(b))))
        checks = {"coefficient_symmetry": bool(residual <= tolerance)}
    result = checked_result("design", derived, method="scipy_firwin" if spec.family == "fir_window" else "scipy_iir_design_sos", trust=trust,
        checks=checks, residual=residual, precision=53, candidate=True,
        details={"family": spec.family, "order": spec.order, "kind": spec.kind,
                 "cutoff_hz": spec.cutoff, "sample_rate": spec.sample_rate,
                 "window": spec.window if spec.family == "fir_window" else None,
                 "passband_ripple_db": spec.passband_ripple, "stopband_attenuation_db": spec.stopband_attenuation,
                 "numeric_pole_radius_max": float(np.max(abs(p), initial=0)),
                 "realization_residual": residual, "verification_absolute_tolerance": tolerance,
                 "specifications_certified": False,
                 "scipy_version": scipy.__version__, "implementation": implementation})
    result.diagnostics.append("Design specifications and stability are numerical candidates; no rigorous bandwise bound is claimed.")
    return result, derived


def cross_spectrum(left, right, *, mode="exact", precision=53, max_exact=64, max_high_precision=256):
    if len(left.samples) != len(right.samples) or sp.simplify(left.sample_rate-right.sample_rate) != 0:
        raise ValueError("cross-spectrum requires equal lengths and sample rates")
    lresult, lspec = dft(left, mode=mode, precision=precision, max_exact=max_exact, max_high_precision=max_high_precision)
    rresult, rspec = dft(right, mode=mode, precision=precision, max_exact=max_exact, max_high_precision=max_high_precision)
    values = tuple(sp.expand(x*sp.conjugate(y)) for x, y in zip(lspec.bins, rspec.bins))
    trust = cap_trust(lresult.trust, rresult.trust)
    derived = Spectrum(bins=values, sample_rate=left.sample_rate, start=left.start-right.start,
        unit=_product_unit(left.unit, right.unit), input_trust=trust)
    return checked_result("cross_spectrum", values, method="fourier_cross_product", trust=trust,
        details={"definition": "DFT(x)*conjugate(DFT(y))", "normalization": "backward",
                 "window": "boxcar", "overlap": 0, "padding": 0, "size": len(values),
                 "sample_rate": left.sample_rate, "frequency_order": "unshifted",
                 "interpretation": "circular_cross_correlation_spectrum; not a PSD estimator",
                 "convention": "exp(-2*pi*i*k*n/N)"}), derived
