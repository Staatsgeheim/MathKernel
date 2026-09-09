# =============================================================================
# MathKernel - engines
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import concurrent.futures
import functools
import math
import os
import pickle
import signal
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
import sympy as sp
from .bigint import decimal_to_int, int_to_decimal


class SolverTimeoutError(ValueError):
    """A solver engine exceeded its configured timeout.

    Subclasses ValueError so existing kernel error handling renders it as a
    structured error result instead of an uncaught exception."""


class IsolatedSolverError(ValueError):
    """A hard-isolated solver failed outside the supported error classes."""


def _isolated_probe(delay_seconds=0.0, fail=False):
    """Small internal health probe used by isolation diagnostics/tests."""
    import time
    time.sleep(delay_seconds)
    if fail:
        raise ValueError("isolated probe failure")
    return os.getpid()


def _isolated_blob(size):
    return b"x"*size


def run_with_timeout(fn, timeout_seconds: float | None, *args, **kwargs):
    """Execute trusted computational Python in a hard-killable process lease.

    Four reusable workers bound concurrency. Execution and queue waits have
    finite timeouts; a cold worker additionally has a 15-second startup limit.
    Unlike the former thread pool, a timeout stops the computation and releases
    capacity. None/zero explicitly selects synchronous trusted-library use.
    """
    from .process_runner import run_isolated_callable
    return run_isolated_callable(fn, timeout_seconds, *args, **kwargs)


def run_in_subprocess(fn, timeout_seconds: float, *args, max_input_bytes=64*1024*1024,
                      max_output_bytes=64*1024*1024, **kwargs):
    """Run an allowlisted MathKernel worker in a fresh, hard-killable process.

    The child may use native threads, but timeout kills its whole process group.
    Only module-level ``_isolated_*`` functions are accepted by the worker.
    """
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("isolated solver timeout must be positive and finite")
    for name, value in (("max_input_bytes", max_input_bytes),
                        ("max_output_bytes", max_output_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not fn.__module__.startswith("mathkernel.") or not fn.__name__.startswith("_isolated_"):
        raise ValueError("isolated workers must be private module-level MathKernel functions")
    request = pickle.dumps({"module": fn.__module__, "function": fn.__name__,
                            "args": args, "kwargs": kwargs,
                            "max_output_bytes": int(max_output_bytes)}, protocol=5)
    if len(request) > max_input_bytes:
        raise IsolatedSolverError("isolated solver request exceeded input limit")
    environment = os.environ.copy()
    # Prevent nested BLAS/OpenMP fan-out inside each isolated solver.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    process = subprocess.Popen([sys.executable, "-m", "mathkernel.isolated_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=environment, start_new_session=os.name != "nt")
    try:
        output, _ = process.communicate(request, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else:
            from .process_runner import terminate_process_tree
            terminate_process_tree(process)
        process.communicate()
        raise SolverTimeoutError(
            f"{fn.__module__}.{fn.__name__} exceeded solver_timeout_seconds={timeout_seconds}; isolated process terminated") from None
    if process.returncode != 0 or not output:
        raise IsolatedSolverError(f"isolated solver exited with code {process.returncode}")
    if len(output) > max_output_bytes:
        raise IsolatedSolverError("isolated solver response exceeded output limit")
    try:
        envelope = pickle.loads(output)
    except Exception as exc:
        raise IsolatedSolverError("isolated solver returned an invalid response") from exc
    if envelope.get("ok"):
        return envelope["value"]
    message = envelope.get("message", "isolated solver failed")
    error_type = envelope.get("error_type")
    exception = {"ImportError": ImportError, "ModuleNotFoundError": ImportError,
                 "NotImplementedError": NotImplementedError, "ValueError": ValueError,
                 "IsolatedSolverError": IsolatedSolverError}.get(error_type, IsolatedSolverError)
    raise exception(message)


def _with_timeout(fn):
    """Run an engine method under the instance's timeout_seconds."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return run_with_timeout(fn, getattr(self, "timeout_seconds", None), self, *args, **kwargs)
    return wrapper

from .models import (AlgebraicNumberNode, BinaryNode, BinderNode, BoolNode, CallNode, ComplexNode, Expr, IntegerNode,
                     IntervalNode, MembershipNode, NaryNode, NumberNode, QuantifierNode, RationalNode,
                     RealNode, RelationNode, SetNode, SetOpNode, SymbolNode, UnaryNode)

_DOMAIN_HINTS = {
    "int": {"integer": True}, "integer": {"integer": True}, "integers": {"integer": True},
    "natural": {"integer": True, "nonnegative": True},
    "positive_integer": {"integer": True, "positive": True},
    "real": {"real": True}, "reals": {"real": True},
    "complex": {}, "complexes": {},
    "positive": {"positive": True}, "negative": {"negative": True},
    "nonnegative": {"nonnegative": True}, "nonpositive": {"nonpositive": True},
    "nonzero": {"nonzero": True},
}
_PROPERTY_HINTS = {
    "positive": {"positive": True}, "negative": {"negative": True},
    "nonnegative": {"nonnegative": True}, "nonpositive": {"nonpositive": True},
    "nonzero": {"nonzero": True},
}


def symbol_env(names, domains: dict[str, str] | None = None,
               properties: dict[str, list[str]] | None = None) -> dict:
    """Build assumption-aware SymPy symbols from context domains/properties.

    Every operation must take its variable from this env: an assumed symbol
    (x with positive=True) is a different object than a bare sp.Symbol('x'),
    and mixing them silently breaks simplification and solving.
    """
    env = {}
    for name in names:
        key = (domains or {}).get(name, "").lower().replace(" ", "_")
        hints = dict(_DOMAIN_HINTS.get(key, {}))
        for prop in (properties or {}).get(name, []):
            hints.update(_PROPERTY_HINTS.get(prop, {}))
        env[name] = sp.Symbol(name, **hints) if hints else sp.Symbol(name)
    return env


class SymPyEngine:
    name = "sympy"
    capabilities = {
        "simplify", "expand", "factor", "cancel", "trig", "rational", "normal_form",
        "solve", "solve_system", "symbolic_equivalence", "evaluate",
        "differentiate", "integrate", "limit", "series", "summation", "product"
    }

    def __init__(self, timeout_seconds: float | None = None):
        self.timeout_seconds = timeout_seconds

    @property
    def available(self) -> bool:
        return True

    def to_sympy(self, node: Expr, env: dict | None = None):
        if isinstance(node, IntegerNode):
            return sp.Integer(decimal_to_int(node.value))
        if isinstance(node, RationalNode):
            return sp.Rational(decimal_to_int(node.numerator), decimal_to_int(node.denominator))
        if isinstance(node, RealNode):
            # ``RealNode.precision`` records the significant decimal digits present
            # in the source literal (e.g. 0.7 -> 1).  Passing that value directly
            # to SymPy Float as the *working* precision catastrophically rounds
            # short literals during symbolic manipulation.  Preserve the literal
            # text, but give SymPy a sane minimum working precision.
            return sp.Float(node.value, max(node.precision or 0, 15))
        if isinstance(node, AlgebraicNumberNode):
            poly_expr = self.to_sympy(node.minimal_polynomial, env)
            names = sorted(poly_expr.free_symbols, key=str)
            if len(names) != 1:
                raise ValueError("Algebraic RootOf requires a univariate minimal polynomial")
            return sp.CRootOf(sp.Poly(poly_expr, names[0]), node.root_index)
        if isinstance(node, ComplexNode):
            return self.to_sympy(node.real, env) + sp.I*self.to_sympy(node.imag, env)
        if isinstance(node, IntervalNode):
            return sp.Interval(self.to_sympy(node.lower, env), self.to_sympy(node.upper, env),
                               left_open=not node.lower_closed, right_open=not node.upper_closed)
        if isinstance(node, NumberNode):
            return sp.Rational(node.value) if "." not in node.value else sp.Float(node.value)
        if isinstance(node, SymbolNode):
            if env is not None and node.name in env:
                return env[node.name]
            return sp.Symbol(node.name)
        if isinstance(node, UnaryNode):
            return -self.to_sympy(node.arg, env)
        if isinstance(node, NaryNode):
            vals = [self.to_sympy(a, env) for a in node.args]
            return sp.Add(*vals) if node.kind == "add" else sp.Mul(*vals)
        if isinstance(node, BinaryNode):
            a, b = self.to_sympy(node.left, env), self.to_sympy(node.right, env)
            return a**b if node.kind == "pow" else a / b
        if isinstance(node, CallNode):
            if node.name == "pi":
                if node.args:
                    raise ValueError("pi takes no arguments")
                return sp.pi
            funcs = {"sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                     "exp": sp.exp, "log": sp.log, "abs": sp.Abs,
                     "factorial": sp.factorial, "gamma": sp.gamma, "binomial": sp.binomial}
            return funcs[node.name](*[self.to_sympy(a, env) for a in node.args])
        if isinstance(node, RelationNode):
            a, b = self.to_sympy(node.left, env), self.to_sympy(node.right, env)
            return {"eq":sp.Eq, "ne":sp.Ne, "lt":sp.Lt, "le":sp.Le,
                    "gt":sp.Gt, "ge":sp.Ge}[node.kind](a, b)
        if isinstance(node, SetNode):
            if node.name is not None:
                return {"naturals": sp.S.Naturals, "integers": sp.S.Integers,
                        "rationals": sp.S.Rationals, "reals": sp.S.Reals,
                        "complexes": sp.S.Complexes, "empty": sp.S.EmptySet}[node.name]
            return sp.FiniteSet(*[self.to_sympy(e, env) for e in node.elements or []])
        if isinstance(node, MembershipNode):
            return sp.Contains(self.to_sympy(node.element, env), self.to_sympy(node.set, env))
        if isinstance(node, SetOpNode):
            args = [self.to_sympy(a, env) for a in node.args]
            if node.op == "union": return sp.Union(*args)
            if node.op == "intersect": return sp.Intersection(*args)
            if node.op == "difference": return sp.Complement(args[0], args[1])
            if len(args) != 2:
                raise ValueError("complement requires an explicit universe: complement(A, universe)")
            return sp.Complement(args[1], args[0])
        if isinstance(node, BoolNode):
            args = [self.to_sympy(a, env) for a in node.args]
            if node.op == "and": return sp.And(*args)
            if node.op == "or": return sp.Or(*args)
            return sp.Not(args[0])
        if isinstance(node, QuantifierNode):
            raise ValueError("SymPy has no quantifier support; use the Z3 engine for quantified statements")
        if isinstance(node, BinderNode):
            domain = node.domain
            if not isinstance(domain, SetNode) or domain.elements is None:
                raise ValueError("SymPy binders require a finite enumerated set domain")
            x = (env or {}).get(node.variable, sp.Symbol(node.variable))
            body = self.to_sympy(node.body, env)
            terms = [body.subs(x, self.to_sympy(e, env)) for e in domain.elements]
            return sp.Add(*terms) if node.op == "sum" else sp.Mul(*terms)
        raise TypeError(type(node))

    def from_sympy(self, expr) -> Expr:
        if isinstance(expr, sp.Symbol): return SymbolNode(name=str(expr))
        if isinstance(expr, sp.Integer): return IntegerNode(value=int_to_decimal(int(expr)))
        if isinstance(expr, sp.Rational):
            if expr.q == 1: return IntegerNode(value=int_to_decimal(int(expr.p)))
            return RationalNode(numerator=int_to_decimal(int(expr.p)), denominator=int_to_decimal(int(expr.q)))
        if isinstance(expr, sp.Float): return RealNode(value=str(expr), precision=int(expr._prec))
        if isinstance(expr, sp.Add): return NaryNode(kind="add", args=[self.from_sympy(a) for a in expr.args])
        if isinstance(expr, sp.Mul): return NaryNode(kind="mul", args=[self.from_sympy(a) for a in expr.args])
        if isinstance(expr, sp.Pow):
            if expr.exp == sp.Rational(1, 2):
                return CallNode(name="sqrt", args=[self.from_sympy(expr.base)])
            if expr.exp == sp.Rational(-1, 2):
                return BinaryNode(kind="div", left=IntegerNode(value="1"),
                                  right=CallNode(name="sqrt", args=[self.from_sympy(expr.base)]))
            return BinaryNode(kind="pow", left=self.from_sympy(expr.base), right=self.from_sympy(expr.exp))
        if isinstance(expr, sp.Equality): return RelationNode(kind="eq", left=self.from_sympy(expr.lhs), right=self.from_sympy(expr.rhs))
        if isinstance(expr, sp.NumberSymbol):
            if expr == sp.E: return CallNode(name="exp", args=[IntegerNode(value="1")])
            if expr == sp.pi: return CallNode(name="pi", args=[])
        funcs={sp.sin:"sin",sp.cos:"cos",sp.tan:"tan",sp.exp:"exp",sp.log:"log",sp.Abs:"abs",
               sp.factorial:"factorial",sp.gamma:"gamma",sp.binomial:"binomial"}
        if expr.func in funcs: return CallNode(name=funcs[expr.func], args=[self.from_sympy(a) for a in expr.args])
        named = {sp.S.Naturals: "naturals", sp.S.Integers: "integers", sp.S.Rationals: "rationals",
                 sp.S.Reals: "reals", sp.S.Complexes: "complexes", sp.S.EmptySet: "empty"}
        for singleton, name in named.items():
            if expr == singleton: return SetNode(name=name)
        if isinstance(expr, sp.FiniteSet):
            return SetNode(elements=[self.from_sympy(e) for e in expr.args])
        if isinstance(expr, sp.Union):
            return SetOpNode(op="union", args=[self.from_sympy(a) for a in expr.args])
        if isinstance(expr, sp.Intersection):
            return SetOpNode(op="intersect", args=[self.from_sympy(a) for a in expr.args])
        if isinstance(expr, sp.Complement):
            return SetOpNode(op="difference", args=[self.from_sympy(a) for a in expr.args])
        if isinstance(expr, sp.Contains):
            return MembershipNode(element=self.from_sympy(expr.args[0]), set=self.from_sympy(expr.args[1]))
        raise ValueError(f"SymPy result cannot yet be represented in MathIR: {type(expr).__name__}")

    def simplify(self, node: Expr, mode: str = "simplify", env: dict | None = None):
        expr = self.to_sympy(node, env)
        fn = {"simplify": sp.simplify, "normal": sp.simplify, "expand": sp.expand,
              "factor": sp.factor, "cancel": sp.cancel, "trig": sp.trigsimp,
              "rational": sp.together,
              "normal_form": lambda e: sp.cancel(sp.together(e))}.get(mode)
        if not fn:
            raise ValueError(f"Unsupported simplify mode: {mode}")
        return fn(expr)

    def solve(self, node: Expr, variable: str, domain: str = "complex", env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        dom = {"real": sp.S.Reals, "integer": sp.S.Integers, "complex": sp.S.Complexes}.get(domain, sp.S.Complexes)
        if isinstance(expr, sp.Equality):
            return sp.solveset(expr.lhs-expr.rhs, x, domain=dom)
        return sp.solveset(expr, x, domain=dom)

    def solve_system(self, nodes: list[Expr], variables: list[str], env: dict | None = None):
        eqs = [self.to_sympy(n, env) for n in nodes]
        syms = [(env or {}).get(v, sp.Symbol(v)) for v in variables]
        return sp.solve(eqs, syms, dict=True)

    def differentiate(self, node: Expr, variable: str, order: int = 1, env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        return sp.diff(expr, x, order).doit()

    def integrate(self, node: Expr, variable: str, lower=None, upper=None, env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        if lower is None and upper is None:
            return sp.integrate(expr, x).doit()
        return sp.integrate(expr, (x, lower, upper)).doit()

    def limit(self, node: Expr, variable: str, point, direction: str = "+-", env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        return sp.limit(expr, x, point, dir=direction).doit()

    def series(self, node: Expr, variable: str, point, order: int = 6, env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        expansion = sp.series(expr, x, point, n=order)
        return expansion.removeO(), expansion.getO() is not None

    def summation(self, node: Expr, variable: str, lower, upper, env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        return sp.summation(expr, (x, lower, upper)).doit()

    def product(self, node: Expr, variable: str, lower, upper, env: dict | None = None):
        expr = self.to_sympy(node, env)
        x = (env or {}).get(variable, sp.Symbol(variable))
        return sp.product(expr, (x, lower, upper)).doit()

    def prove_equivalence(self, left: Expr, right: Expr, env: dict | None = None):
        diff = sp.simplify(self.to_sympy(left, env) - self.to_sympy(right, env))
        return diff == 0, diff


for _timed in ("simplify", "solve", "solve_system", "differentiate", "integrate",
               "limit", "series", "summation", "product", "prove_equivalence"):
    setattr(SymPyEngine, _timed, _with_timeout(getattr(SymPyEngine, _timed)))
del _timed


class Z3Engine:
    """Optional exact SMT checker for the arithmetic fragment supported by MathIR."""

    name = "z3"
    capabilities = {"smt", "counterexample", "real_arithmetic", "integer_arithmetic", "equivalence_check"}

    def __init__(self, timeout_ms: int | None = None):
        self.timeout_ms = timeout_ms
        try:
            import z3  # type: ignore
            self.z3 = z3
            self._available = True
        except ImportError:
            self.z3 = None
            self._available = False

    def _new_solver(self):
        solver = self.z3.Solver()
        if self.timeout_ms:
            solver.set(timeout=self.timeout_ms)
        return solver

    @property
    def available(self) -> bool:
        return self._available

    def _sort_for(self, symbol: str, domains: dict[str, str]):
        return "Int" if domains.get(symbol, "real").lower() in {"int", "integer", "integers"} else "Real"

    def _symbols(self, node: Expr, out: set[str] | None = None, bound: set[str] | None = None) -> set[str]:
        if out is None: out = set()
        if bound is None: bound = set()
        if isinstance(node, SymbolNode):
            if node.name not in bound: out.add(node.name)
        elif isinstance(node, UnaryNode): self._symbols(node.arg, out, bound)
        elif isinstance(node, NaryNode):
            for a in node.args: self._symbols(a, out, bound)
        elif isinstance(node, BinaryNode): self._symbols(node.left, out, bound); self._symbols(node.right, out, bound)
        elif isinstance(node, CallNode):
            raise ValueError(f"Z3 fragment does not support function {node.name}")
        elif isinstance(node, RelationNode): self._symbols(node.left, out, bound); self._symbols(node.right, out, bound)
        elif isinstance(node, SetNode):
            for e in node.elements or []: self._symbols(e, out, bound)
        elif isinstance(node, MembershipNode):
            self._symbols(node.element, out, bound); self._symbols(node.set, out, bound)
        elif isinstance(node, SetOpNode):
            for a in node.args: self._symbols(a, out, bound)
        elif isinstance(node, BoolNode):
            for a in node.args: self._symbols(a, out, bound)
        elif isinstance(node, QuantifierNode):
            if node.domain is not None: self._symbols(node.domain, out, bound)
            self._symbols(node.body, out, bound | {node.variable})
        elif isinstance(node, BinderNode):
            raise ValueError("Z3 fragment does not support sum/product binders")
        return out

    def _env(self, nodes: list[Expr], domains: dict[str, str]):
        z3 = self.z3
        names: set[str] = set()
        for n in nodes: self._symbols(n, names)
        return {n: (z3.Int(n) if self._sort_for(n, domains) == "Int" else z3.Real(n)) for n in names}

    def to_z3(self, node: Expr, env: dict):
        z3 = self.z3
        if isinstance(node, IntegerNode): return z3.IntVal(node.value)
        if isinstance(node, RationalNode): return z3.RealVal(f"{node.numerator}/{node.denominator}")
        if isinstance(node, RealNode): return z3.RealVal(node.value)
        if isinstance(node, NumberNode): return z3.RealVal(node.value)
        if isinstance(node, SymbolNode): return env[node.name]
        if isinstance(node, UnaryNode): return -self.to_z3(node.arg, env)
        if isinstance(node, NaryNode):
            vals = [self.to_z3(a, env) for a in node.args]
            if node.kind == "add": return sum(vals)
            out = vals[0]
            for v in vals[1:]: out = out * v
            return out
        if isinstance(node, BinaryNode):
            a, b = self.to_z3(node.left, env), self.to_z3(node.right, env)
            if node.kind == "div": return a / b
            if not isinstance(node.right, (IntegerNode, NumberNode)):
                raise ValueError("Z3 prototype only supports literal integer exponents")
            exponent = decimal_to_int(node.right.value)
            if exponent < 0: raise ValueError("Z3 prototype does not support negative exponents")
            return a ** exponent
        if isinstance(node, RelationNode):
            a, b = self.to_z3(node.left, env), self.to_z3(node.right, env)
            return {"eq":lambda: a==b, "ne":lambda:a!=b, "lt":lambda:a<b,
                    "le":lambda:a<=b, "gt":lambda:a>b, "ge":lambda:a>=b}[node.kind]()
        if isinstance(node, MembershipNode):
            element = self.to_z3(node.element, env)
            return self._membership(node.set, element, env)
        if isinstance(node, BoolNode):
            args = [self.to_z3(a, env) for a in node.args]
            if node.op == "and": return z3.And(*args)
            if node.op == "or": return z3.Or(*args)
            return z3.Not(args[0])
        if isinstance(node, QuantifierNode):
            return self._quantifier_z3(node, env)
        if isinstance(node, (SetNode, SetOpNode)):
            raise ValueError("Set values are not Z3 expressions; use them inside in(...), forall(...) or exists(...)")
        if isinstance(node, BinderNode):
            raise ValueError("Z3 fragment does not support sum/product binders")
        if isinstance(node, CallNode): raise ValueError(f"Z3 fragment does not support {node.name}")
        raise TypeError(type(node))

    def _membership(self, set_node: Expr, element, env: dict):
        """Z3 predicate: element ∈ set_node, for the supported set fragment."""
        z3 = self.z3
        if isinstance(set_node, SetOpNode):
            parts = [self._membership(a, element, env) for a in set_node.args]
            if set_node.op == "union": return z3.Or(*parts)
            if set_node.op == "intersect": return z3.And(*parts)
            if set_node.op == "difference": return z3.And(parts[0], z3.Not(parts[1]))
            if len(parts) == 2: return z3.And(parts[1], z3.Not(parts[0]))  # complement(A, universe)
            raise ValueError("complement requires an explicit universe in the Z3 fragment")
        if not isinstance(set_node, SetNode):
            raise ValueError("Z3 membership requires a set literal, named set, or set operation")
        if set_node.elements is not None:
            if not set_node.elements: return z3.BoolVal(False)
            return z3.Or(*[element == self.to_z3(e, env) for e in set_node.elements])
        name = set_node.name
        if name == "empty": return z3.BoolVal(False)
        if name == "naturals": return element >= 0
        if name == "integers": return z3.BoolVal(True)  # sort constraint is enforced separately
        if name in ("reals", "rationals"): return z3.BoolVal(True)
        raise ValueError(f"Z3 membership does not support the set {name}")

    def _quantifier_z3(self, node: QuantifierNode, env: dict):
        z3 = self.z3
        domain = node.domain
        sort_int = isinstance(domain, SetNode) and (
            domain.name in ("naturals", "integers") or
            (domain.elements is not None and all(isinstance(e, IntegerNode) for e in domain.elements)))
        if isinstance(domain, SetNode) and domain.name == "complexes":
            raise ValueError("Z3 quantifiers do not support complex domains")
        bound = z3.Int(node.variable) if sort_int else z3.Real(node.variable)
        local_env = {**env, node.variable: bound}
        body = self.to_z3(node.body, local_env)
        if domain is not None:
            guard = self._membership(domain, bound, env)
            body = z3.Implies(guard, body) if node.quantifier == "forall" else z3.And(guard, body)
        return z3.ForAll([bound], body) if node.quantifier == "forall" else z3.Exists([bound], body)

    def counterexample_equivalence(self, left: Expr, right: Expr, assumptions: list[Expr], domains: dict[str,str]):
        if not self.available: raise RuntimeError("z3-solver is not installed")
        z3 = self.z3
        env = self._env([left, right, *assumptions], domains)
        solver = self._new_solver()
        for a in assumptions: solver.add(self.to_z3(a, env))
        solver.add(self.to_z3(left, env) != self.to_z3(right, env))
        answer = solver.check()
        if answer == z3.unsat: return "proved", None
        if answer == z3.sat:
            model = solver.model()
            return "disproved", {name: str(model.eval(sym, model_completion=True)) for name, sym in env.items()}
        return "unknown", None

    def check_context(self, assumptions: list[Expr], domains: dict[str,str]):
        if not self.available: raise RuntimeError("z3-solver is not installed")
        z3 = self.z3
        env = self._env(assumptions, domains)
        s = self._new_solver()
        for a in assumptions: s.add(self.to_z3(a, env))
        r = s.check()
        return "consistent" if r == z3.sat else ("inconsistent" if r == z3.unsat else "unknown")

    def check_relation(self, relation: Expr, assumptions: list[Expr], domains: dict[str,str]):
        """Check whether a relation is satisfiable under the current context."""
        if not self.available: raise RuntimeError("z3-solver is not installed")
        if not isinstance(relation, RelationNode): raise ValueError("Expected a relation")
        z3 = self.z3
        env = self._env([relation, *assumptions], domains)
        s = self._new_solver()
        for a in assumptions: s.add(self.to_z3(a, env))
        s.add(self.to_z3(relation, env))
        r = s.check()
        if r == z3.sat:
            m=s.model()
            return "sat", {name:str(m.eval(sym, model_completion=True)) for name,sym in env.items()}
        if r == z3.unsat: return "unsat", None
        return "unknown", None

    def find_solution_outside_candidates(self, relation: Expr, variable: str, candidates: list,
                                         assumptions: list[Expr], domains: dict[str,str], sympy_engine):
        """Search for a satisfying value not present in a finite candidate set.

        This is an independent completeness check for the single-variable arithmetic
        fragment. Candidates are generated internally by the symbolic backend and
        converted through MathIR, never by evaluating user strings.
        """
        if not self.available: raise RuntimeError("z3-solver is not installed")
        if not isinstance(relation, RelationNode) or relation.kind != "eq":
            raise ValueError("Completeness checking currently requires an equality")
        z3=self.z3
        env=self._env([relation,*assumptions],domains)
        if variable not in env: raise ValueError(f"Variable {variable} does not occur in relation")
        solver = self._new_solver()
        for a in assumptions: solver.add(self.to_z3(a,env))
        solver.add(self.to_z3(relation,env))
        for candidate in candidates:
            try:
                candidate_ir=sympy_engine.from_sympy(candidate)
                candidate_z3=self.to_z3(candidate_ir,env)
            except (ValueError,TypeError) as exc:
                raise ValueError(f"Candidate {candidate} is outside the current Z3 conversion fragment: {exc}") from exc
            solver.add(env[variable] != candidate_z3)
        r=solver.check()
        if r == z3.unsat: return "proved", None
        if r == z3.sat:
            m=solver.model()
            return "disproved", {name:str(m.eval(sym,model_completion=True)) for name,sym in env.items()}
        return "unknown", None


class LeanEngine:
    """Lean 4 certificate generator/checker with semantic tactic selection for a safe arithmetic subset."""

    name = "lean"
    capabilities = {"formal_ring_equivalence", "formal_certificate", "formal_linarith", "formal_nlinarith", "formal_norm_num", "formal_omega"}

    def __init__(self, executable: str = "lean", timeout: float = 90.0):
        self.executable = executable
        self.timeout = timeout

    @property
    def available(self) -> bool:
        from .lean_bootstrap import resolve_lean_toolchain
        return resolve_lean_toolchain() is not None

    def _symbols(self, node: Expr, out: set[str] | None = None) -> set[str]:
        if out is None: out = set()
        if isinstance(node, SymbolNode): out.add(node.name)
        elif isinstance(node, UnaryNode): self._symbols(node.arg, out)
        elif isinstance(node, NaryNode):
            for a in node.args: self._symbols(a, out)
        elif isinstance(node, BinaryNode): self._symbols(node.left, out); self._symbols(node.right, out)
        elif isinstance(node, RelationNode): self._symbols(node.left, out); self._symbols(node.right, out)
        elif isinstance(node, CallNode): raise ValueError("Lean arithmetic fragment does not support functions")
        return out

    def to_lean(self, node: Expr) -> str:
        if isinstance(node, IntegerNode): return node.value
        if isinstance(node, RationalNode): return f"({node.numerator} / {node.denominator})"
        if isinstance(node, RealNode):
            # finite decimals are exactly rational; emit them as fractions
            frac = Fraction(node.value)
            return f"({frac.numerator} / {frac.denominator})"
        if isinstance(node, NumberNode):
            if "." in node.value:
                frac = Fraction(node.value)
                return f"({frac.numerator} / {frac.denominator})"
            return node.value
        if isinstance(node, SymbolNode): return node.name
        if isinstance(node, UnaryNode): return f"(-{self.to_lean(node.arg)})"
        if isinstance(node, NaryNode):
            op = " + " if node.kind == "add" else " * "
            return "(" + op.join(self.to_lean(a) for a in node.args) + ")"
        if isinstance(node, BinaryNode):
            if node.kind == "div":
                # division stays in the fragment only by a nonzero literal
                if node.right.kind in ("integer", "number") and int(node.right.value) != 0:
                    return f"({self.to_lean(node.left)} / {node.right.value})"
                if node.right.kind == "rational":
                    return f"({self.to_lean(node.left)} / ({node.right.numerator} / {node.right.denominator}))"
                raise ValueError("Lean arithmetic fragment supports division only by nonzero literals")
            if not isinstance(node.right, (IntegerNode, NumberNode)):
                raise ValueError("Lean arithmetic fragment requires natural-number literal powers")
            exponent=int(node.right.value)
            if exponent < 0: raise ValueError("Lean arithmetic fragment requires non-negative powers")
            return f"({self.to_lean(node.left)} ^ {exponent})"
        if isinstance(node, RelationNode):
            op = {"eq":"=", "ne":"≠", "lt":"<", "le":"≤", "gt":">", "ge":"≥"}[node.kind]
            return f"({self.to_lean(node.left)} {op} {self.to_lean(node.right)})"
        if isinstance(node, CallNode): raise ValueError("Lean arithmetic fragment does not support functions")
        raise TypeError(type(node))

    def _max_degree(self, node: Expr) -> int:
        if isinstance(node, (IntegerNode, RationalNode, RealNode, NumberNode)): return 0
        if isinstance(node, SymbolNode): return 1
        if isinstance(node, UnaryNode): return self._max_degree(node.arg)
        if isinstance(node, NaryNode):
            ds=[self._max_degree(a) for a in node.args]
            return max(ds, default=0) if node.kind == "add" else sum(ds)
        if isinstance(node, BinaryNode):
            if node.kind == "div": raise ValueError("division is outside polynomial tactic selection")
            if not isinstance(node.right, (IntegerNode, NumberNode)): raise ValueError("non-literal power")
            return self._max_degree(node.left) * decimal_to_int(node.right.value)
        if isinstance(node, RelationNode): return max(self._max_degree(node.left), self._max_degree(node.right))
        raise ValueError("non-polynomial expression")

    def select_tactic(self, left: Expr, right: Expr, assumptions: list[Expr] | None = None,
                      integer: bool = False) -> str:
        assumptions = assumptions or []
        names=self._symbols(left)|self._symbols(right)
        for a in assumptions: names |= self._symbols(a)
        if not names: return "norm_num"
        degree=max([self._max_degree(left), self._max_degree(right), *[self._max_degree(a) for a in assumptions]])
        if integer and degree <= 1: return "omega"
        if assumptions: return "linarith" if degree <= 1 else "nlinarith"
        return "ring" if not integer else ("nlinarith" if degree > 1 else "omega")

    def build_certificate(self, left: Expr, right: Expr, assumptions: list[Expr] | None = None,
                          integer: bool = False) -> tuple[str, str]:
        assumptions = assumptions or []
        names = self._symbols(left) | self._symbols(right)
        for a in assumptions: names |= self._symbols(a)
        sort = "ℤ" if integer else "ℚ"
        binders = " ".join(f"({n} : {sort})" for n in sorted(names))
        hypotheses = " ".join(f"(h{i} : {self.to_lean(a)})" for i,a in enumerate(assumptions))
        tactic = self.select_tactic(left, right, assumptions, integer=integer)
        pieces=[x for x in [binders,hypotheses] if x]
        prefix=(" " + " ".join(pieces)) if pieces else ""
        script=("import Mathlib\n\n" + f"example{prefix} : {self.to_lean(left)} = {self.to_lean(right)} := by\n" + f"  {tactic}\n")
        return script, tactic

    def build_ring_certificate(self, left: Expr, right: Expr) -> str:
        script, _ = self.build_certificate(left, right, [])
        return script

    def build_relation_certificate(self, relation: RelationNode,
                                   assumptions: list[Expr] | None = None,
                                   integer: bool = False) -> tuple[str, str]:
        """Certificate for a general relation goal (not just equalities)."""
        assumptions = assumptions or []
        if relation.kind == "eq":
            return self.build_certificate(relation.left, relation.right, assumptions, integer)
        names = self._symbols(relation)
        for a in assumptions:
            names |= self._symbols(a)
        sort = "ℤ" if integer else "ℚ"
        binders = " ".join(f"({n} : {sort})" for n in sorted(names))
        hypotheses = " ".join(f"(h{i} : {self.to_lean(a)})" for i, a in enumerate(assumptions))
        degree = max([self._max_degree(relation.left), self._max_degree(relation.right),
                      *[self._max_degree(a) for a in assumptions]])
        if not names:
            tactic = "norm_num"
        elif integer and degree <= 1:
            tactic = "omega"
        else:
            tactic = "linarith" if degree <= 1 else "nlinarith"
        pieces = [x for x in [binders, hypotheses] if x]
        prefix = (" " + " ".join(pieces)) if pieces else ""
        script = ("import Mathlib\n\n" +
                  f"example{prefix} : {self.to_lean(relation)} := by\n  {tactic}\n")
        return script, tactic

    def prove_relation(self, relation: RelationNode, assumptions: list[Expr] | None = None,
                       integer: bool = False, timeout: float | None = None):
        """Prove a relation goal; returns (status, script, tactic, error)."""
        self._require_exact_inputs([relation, *(assumptions or [])])
        if timeout is None:
            timeout = self.timeout
        script, tactic = self.build_relation_certificate(relation, assumptions, integer)
        return self._check_script(script, tactic, timeout)

    def prove_equivalence(self, left: Expr, right: Expr, assumptions: list[Expr] | None = None, timeout: float | None = None):
        self._require_exact_inputs([left, right, *(assumptions or [])])
        if timeout is None:
            timeout = self.timeout
        script, tactic = self.build_certificate(left, right, assumptions)
        return self._check_script(script, tactic, timeout)

    @staticmethod
    def _require_exact_inputs(expressions: list[Expr]) -> None:
        from .reasoning import _walk
        kinds: set[str] = set()
        for expression in expressions:
            _walk(expression, set(), set(), kinds)
        if "real" in kinds:
            raise ValueError("Formal proofs are disabled for approximate decimal inputs or assumptions")

    def prove_ring_equivalence(self, left: Expr, right: Expr, timeout: float | None = None):
        status, script, _tactic, error = self.prove_equivalence(left, right, [], timeout)
        return status, script, error

    def tactic_coverage(self) -> dict:
        """Which MathIR fragment each selectable tactic is expected to decide.
        This is a coverage map for introspection, not a completeness theorem."""
        return {
            "norm_num": "closed rational arithmetic goals (no free symbols)",
            "ring": "polynomial identities over commutative rings, no assumptions, "
                    "natural-number literal powers, division by nonzero literals",
            "linarith": "linear rational arithmetic with linear assumptions (degree <= 1)",
            "nlinarith": "nonlinear rational arithmetic with polynomial assumptions "
                         "(best-effort; not a decision procedure)",
            "omega": "linear integer arithmetic (Presburger) goals and assumptions "
                     "over ℤ; a decision procedure for that fragment",
        }

    def replay_certificate(self, script: str, timeout: float | None = None) -> dict:
        """Deterministically re-check a previously generated Lean certificate.
        The script is treated as opaque text produced by build_certificate."""
        if timeout is None:
            timeout = self.timeout
        status, _script, _tactic, error = self._check_script(script, "replay", timeout)
        if status == "proved":
            return {"status": "proved"}
        if status == "unavailable":
            return {"status": "unavailable", "error": error}
        return {"status": "error", "error": error}

    def _check_script(self, script: str, tactic: str, timeout: float):
        from .lean_bootstrap import run_lean_script
        if not self.available:
            return "unavailable", script, tactic, "Lean/Mathlib is unavailable; run mathkernel-lean-setup explicitly to install or repair it"
        try:
            p = run_lean_script(script, timeout)
        except OSError as exc:
            return "unavailable", script, tactic, str(exc)
        except subprocess.TimeoutExpired:
            return "error", script, tactic, "Lean proof timed out"
        if p.returncode == 0:
            return "proved", script, tactic, None
        return "error", script, tactic, (p.stderr or p.stdout)[-4000:]
