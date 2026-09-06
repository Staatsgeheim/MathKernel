# =============================================================================
# MathKernel - integers
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import math

import sympy as sp

from .bigint import decimal_digit_count, decimal_to_int, int_to_decimal
from .parallel import process_map


def _integer_job(job: dict) -> dict:
    """Module-level worker so batches are picklable under the spawn start method."""
    engine = IntegerEngine()
    try:
        data = engine.compute(
            job["operation"], job.get("values", []),
            modulus=job.get("modulus"), moduli=job.get("moduli"),
            max_output_digits=int(job.get("max_output_digits", 100_000)),
            factor_limit=int(job.get("factor_limit", 100_000)),
        )
        return {"ok": True, "operation": job["operation"], "data": data}
    except (ValueError, OverflowError, KeyError) as exc:
        return {"ok": False, "operation": job.get("operation"), "error": str(exc)}


class IntegerEngine:
    """Arbitrary-precision integer and elementary number-theory engine.

    Decimal parsing/rendering intentionally avoids CPython's int string digit limit.
    Expensive factorization is bounded by SymPy's trial-factor limit in this prototype.
    """
    name = "integer_exact"
    capabilities = {
        "big_integer", "integer_analyze", "integer_batch", "gcd", "lcm", "is_prime",
        "factor_integer", "factorial", "binomial", "mod", "mod_inverse", "pow_mod", "crt",
        "affine_jump"
    }

    @property
    def available(self) -> bool:
        return True

    def _encoded(self, value: int, max_output_digits: int, representation: str | None = None) -> dict:
        digits = decimal_digit_count(value)
        out = {
            "decimal_digits": digits,
            "bit_length": abs(value).bit_length(),
            "exact": True,
            "truncated": digits > max_output_digits,
        }
        if digits <= max_output_digits:
            out["value"] = int_to_decimal(value)
        else:
            out["value"] = None
            out["representation"] = representation
            out["reason"] = f"Exact result has {digits} decimal digits; output limit is {max_output_digits}."
        return out

    def analyze(self, value: str, factor_limit: int = 100_000) -> dict:
        n = decimal_to_int(value)
        mag = abs(n)
        bits = mag.bit_length()
        primality_checked = bits <= 8192
        data = {
            "sign": -1 if n < 0 else (1 if n > 0 else 0),
            "decimal_digits": len(value.lstrip("+-").lstrip("0")) or 1,
            "bit_length": bits,
            "is_even": n % 2 == 0,
            "is_prime": (bool(sp.isprime(mag)) if mag >= 2 and primality_checked else (False if mag < 2 else None)),
            "primality_checked": primality_checked,
            "fits_int64": -(2**63) <= n <= 2**63-1,
        }
        if not primality_checked:
            data["primality_note"] = "Automatic primality testing skipped above 8192 bits; request is_prime explicitly."
        # Cheap bounded factor preview only for modest values.
        if mag >= 2 and mag.bit_length() <= 512:
            f = sp.factorint(mag, limit=factor_limit)
            data["factorization"] = {int_to_decimal(int(p)): int(e) for p, e in f.items()}
            data["factorization_complete"] = all(sp.isprime(int(p)) for p in f)
        return data

    def compute(self, operation: str, values: list[str], *, modulus: str | None = None,
                moduli: list[str] | None = None, max_output_digits: int = 100_000,
                factor_limit: int = 100_000) -> dict:
        if not values and operation != "crt":
            raise ValueError("values must not be empty")
        nums = [decimal_to_int(v) for v in values]
        op = operation.lower()
        if op == "add":
            return {"operation": op, "result": self._encoded(sum(nums), max_output_digits)}
        if op == "sub":
            r = nums[0]
            for n in nums[1:]: r -= n
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "mul":
            r = 1
            for n in nums: r *= n
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "div":
            if len(nums) != 2: raise ValueError("div expects two values")
            if nums[1] == 0: raise ValueError("division by zero")
            from fractions import Fraction
            q = Fraction(nums[0], nums[1])
            text = int_to_decimal(q.numerator) if q.denominator == 1 \
                else f"{int_to_decimal(q.numerator)}/{int_to_decimal(q.denominator)}"
            return {"operation": op, "result": text, "exact_rational": q.denominator != 1}
        if op == "pow":
            if len(nums) != 2: raise ValueError("pow expects base and exponent")
            base, exp = nums
            if exp < 0: raise ValueError("pow expects a non-negative exponent; use div for reciprocals")
            if base in (0, 1, -1):
                r = 0 if base == 0 and exp > 0 else base ** exp
            else:
                if exp * abs(base).bit_length() > max_output_digits * 4:
                    raise ValueError(
                        f"pow result would exceed {max_output_digits} digits; raise max_output_digits")
                r = base ** exp
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "next_prime":
            if len(nums) != 1: raise ValueError("next_prime expects one value")
            return {"operation": op, "result": self._encoded(int(sp.nextprime(nums[0])), max_output_digits)}
        if op == "prev_prime":
            if len(nums) != 1: raise ValueError("prev_prime expects one value")
            if nums[0] <= 2: raise ValueError("no prime below or at 2")
            return {"operation": op, "result": self._encoded(int(sp.prevprime(nums[0])), max_output_digits)}
        if op == "fibonacci":
            if len(nums) != 1 or nums[0] < 0: raise ValueError("fibonacci expects one non-negative integer")
            if nums[0] > 1_000_000: raise ValueError("prototype fibonacci resource limit is n <= 1000000")
            return {"operation": op, "result": self._encoded(int(sp.fibonacci(nums[0])), max_output_digits)}
        if op == "gcd":
            r = 0
            for n in nums: r = math.gcd(r, n)
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "lcm":
            r = 1
            for n in nums: r = math.lcm(r, n)
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "is_prime":
            if len(nums) != 1: raise ValueError("is_prime expects one value")
            return {"operation": op, "is_prime": bool(sp.isprime(abs(nums[0])))}
        if op == "factor":
            if len(nums) != 1: raise ValueError("factor expects one value")
            n = abs(nums[0])
            if n < 2: return {"operation": op, "factors": {}, "complete": True}
            f = sp.factorint(n, limit=factor_limit)
            return {"operation": op,
                    "factors": {int_to_decimal(int(p)): int(e) for p, e in f.items()},
                    "complete": all(sp.isprime(int(p)) for p in f),
                    "factor_limit": factor_limit}
        if op == "factorial":
            if len(nums) != 1 or nums[0] < 0: raise ValueError("factorial expects one non-negative integer")
            n = nums[0]
            if n > 200_000: raise ValueError("prototype factorial resource limit is n <= 200000")
            r = math.factorial(n)
            return {"operation": op, "result": self._encoded(r, max_output_digits, f"{int_to_decimal(n)}!")}
        if op == "binomial":
            if len(nums) != 2: raise ValueError("binomial expects n and k")
            n, k = nums
            if n < 0 or k < 0: raise ValueError("prototype binomial expects non-negative n and k")
            if n > 1_000_000: raise ValueError("prototype binomial resource limit is n <= 1000000")
            r = math.comb(n, k)
            return {"operation": op, "result": self._encoded(r, max_output_digits, f"C({n},{k})")}
        if op == "mod":
            if len(nums) != 1 or modulus is None: raise ValueError("mod expects one value and modulus")
            m = decimal_to_int(modulus)
            if m == 0: raise ValueError("modulus cannot be zero")
            return {"operation": op, "result": self._encoded(nums[0] % m, max_output_digits)}
        if op == "mod_inverse":
            if len(nums) != 1 or modulus is None: raise ValueError("mod_inverse expects one value and modulus")
            m = decimal_to_int(modulus)
            r = pow(nums[0], -1, m)
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "pow_mod":
            if len(nums) != 2 or modulus is None: raise ValueError("pow_mod expects base, exponent and modulus")
            m = decimal_to_int(modulus)
            if m == 0: raise ValueError("modulus cannot be zero")
            r = pow(nums[0], nums[1], m)
            return {"operation": op, "result": self._encoded(r, max_output_digits)}
        if op == "crt":
            if not moduli or len(moduli) != len(nums): raise ValueError("crt expects equally sized values and moduli")
            mods = [decimal_to_int(m) for m in moduli]
            from sympy.ntheory.modular import crt
            result = crt(mods, nums, check=True)
            if result is None: return {"operation": op, "consistent": False, "result": None}
            r, m = map(int, result)
            return {"operation": op, "consistent": True,
                    "result": self._encoded(r, max_output_digits),
                    "modulus": self._encoded(m, max_output_digits)}
        if op == "affine_jump":
            if len(nums) != 3 or modulus is None:
                raise ValueError("affine_jump expects values [multiplier, increment, k] and a modulus")
            a, c, k = nums
            m = decimal_to_int(modulus)
            if m <= 0: raise ValueError("modulus must be positive")
            if k < 0: raise ValueError("k must be non-negative")
            # T^k(s) = a^k * s + c * (1 + a + ... + a^(k-1))  (mod m).
            # The geometric sum is evaluated by doubling because a - 1 need
            # not be invertible mod m (e.g. m = 2^64 with a odd => a - 1 even).
            ak, sk = 1 % m, 0
            cur_a, cur_s = a % m, 1 % m  # a^(2^j), sum_{i<2^j} a^i
            while k:
                if k & 1:
                    sk = (sk + ak * cur_s) % m
                    ak = (ak * cur_a) % m
                cur_s = (cur_s * (1 + cur_a)) % m
                cur_a = (cur_a * cur_a) % m
                k >>= 1
            return {"operation": op,
                    "power": self._encoded(ak, max_output_digits),
                    "translation": self._encoded((c % m) * sk % m, max_output_digits),
                    "modulus": self._encoded(m, max_output_digits)}
        raise ValueError(f"unsupported integer operation: {operation}")

    def batch(self, jobs: list[dict], *, workers: int = 1) -> list[dict]:
        """Run independent compute jobs across worker processes, preserving order.

        Pools are persistent (see parallel.process_map), so repeated batches
        do not repay interpreter spawn costs. For bulk 64-bit modular work
        inside scan loops, use the array-level kernels in integers_fast
        directly; the dict-shaped per-job API here is glue-bound, not
        compute-bound, so compiling it does not pay.
        """
        return process_map(_integer_job, jobs, workers=workers)
