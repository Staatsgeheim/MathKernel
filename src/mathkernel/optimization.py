# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Typed LP/QP/MILP candidates with independent original-data certificate checks.

Canonical constraints are G*x <= h, A*x = b. Bounds are appended to G.
For minimization: Q*x+c+G.T*lambda+A.T*nu=0, lambda>=0.
A solver's termination status never certifies global optimality.
"""
from __future__ import annotations
from typing import Literal
import sympy as sp
from pydantic import model_validator
from .engineering import (EngineeringModel, EngineeringResult, arithmetic_trust,
    cap_trust, checked_result, numeric_array, validate_scalars)
from mathkernel_artifacts import CertificateEvidence, ComputationEvidence, EvidenceBundle, NumericalEvidence


class OptimizationProblem(EngineeringModel):
    variables: tuple[str, ...]
    c: tuple[sp.Expr, ...]
    Q: tuple[tuple[sp.Expr, ...], ...] = ()
    A_ub: tuple[tuple[sp.Expr, ...], ...] = ()
    b_ub: tuple[sp.Expr, ...] = ()
    A_eq: tuple[tuple[sp.Expr, ...], ...] = ()
    b_eq: tuple[sp.Expr, ...] = ()
    lower: tuple[sp.Expr | None, ...] = ()  # omitted means x >= 0
    upper: tuple[sp.Expr | None, ...] = ()
    integrality: tuple[Literal[0, 1], ...] = ()
    sense: Literal["min", "max"] = "min"
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_problem(self):
        n = len(self.c)
        if not n or len(self.variables) != n or len(set(self.variables)) != n:
            raise ValueError("variables must be unique and match objective dimension")
        for matrix, rhs in ((self.A_ub, self.b_ub), (self.A_eq, self.b_eq)):
            if len(matrix) != len(rhs) or any(len(row) != n for row in matrix):
                raise ValueError("constraint dimensions do not match objective")
        if self.Q and (len(self.Q) != n or any(len(row) != n for row in self.Q)):
            raise ValueError("Q must be square with objective dimension")
        if self.Q and any(self.Q[i][j] != self.Q[j][i] for i in range(n) for j in range(n)):
            raise ValueError("Q must be explicitly symmetric")
        for seq in (self.lower, self.upper, self.integrality):
            if seq and len(seq) != n:
                raise ValueError("bound/integrality dimensions do not match objective")
        if self.Q and any(self.integrality):
            raise ValueError("mixed-integer quadratic optimization is not implemented")
        values = list(self.c)+list(self.b_ub)+list(self.b_eq)
        values += [x for m in (self.Q, self.A_ub, self.A_eq) for row in m for x in row]
        values += [x for seq in (self.lower, self.upper) for x in seq if x is not None]
        validate_scalars(values, real=True)
        if any(x.free_symbols for x in values):
            raise ValueError("optimization currently requires concrete real coefficients")
        cap_trust(self.input_trust)
        return self

    @property
    def family(self):
        return "milp" if any(self.integrality) else "qp" if self.Q else "lp"


class OptimizationCertificate(EngineeringModel):
    kind: Literal["optimal", "infeasible", "unbounded"] = "optimal"
    primal: tuple[sp.Expr, ...] = ()
    ray: tuple[sp.Expr, ...] = ()
    inequality_dual: tuple[sp.Expr, ...] = ()
    equality_dual: tuple[sp.Expr, ...] = ()

    @model_validator(mode="after")
    def validate_certificate(self):
        validate_scalars((*self.primal, *self.inequality_dual, *self.equality_dual, *self.ray), real=True)
        return self


def canonical(problem):
    n = len(problem.c)
    sign = 1 if problem.sense == "min" else -1
    Q = sign*sp.Matrix(problem.Q) if problem.Q else sp.zeros(n)
    c = sign*sp.Matrix(problem.c)
    rows, rhs = list(problem.A_ub), list(problem.b_ub)
    lower = problem.lower or (sp.S.Zero,)*n
    upper = problem.upper or (None,)*n
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if lo is not None:
            row = [sp.S.Zero]*n
            row[i] = -sp.S.One
            rows.append(tuple(row))
            rhs.append(-lo)
        if hi is not None:
            row = [sp.S.Zero]*n
            row[i] = sp.S.One
            rows.append(tuple(row))
            rhs.append(hi)
    G = sp.Matrix(rows) if rows else sp.zeros(0, n)
    A = sp.Matrix(problem.A_eq) if problem.A_eq else sp.zeros(0, n)
    return Q, c, G, sp.Matrix(len(rhs), 1, rhs), A, sp.Matrix(len(problem.b_eq), 1, problem.b_eq)


def exact_psd(matrix):
    """Exact symmetric Schur-complement PSD test, including singular matrices.

    Zero diagonal with a nonzero off-diagonal cannot be PSD. Positive pivots
    reduce the remaining Schur complement; no eigenvalue tolerance is used.
    """
    work = matrix.copy()
    pivots = []
    while work.rows:
        diagonal = [work[i, i] for i in range(work.rows)]
        if any(x.is_negative is True for x in diagonal):
            return False, pivots
        pivot = next((i for i, x in enumerate(diagonal) if x.is_positive is True), None)
        if pivot is None:
            if all(x == 0 for x in diagonal):
                return all(x == 0 for x in work), pivots
            return None, pivots
        if pivot:
            work.row_swap(0, pivot)
            work.col_swap(0, pivot)
        d = work[0, 0]
        pivots.append(d)
        work = (work[1:, 1:]-work[1:, :1]*work[:1, 1:]/d).applyfunc(sp.cancel)
    return True, pivots


def _problem_trust(problem):
    matrices = canonical(problem)
    return arithmetic_trust((x for m in matrices for x in m), problem.input_trust)


def verify_certificate(problem, certificate, *, certificate_trust="exact"):
    if certificate.kind != "optimal":
        from .optimization_proofs import verify_outcome
        return verify_outcome(problem, certificate, certificate_trust=certificate_trust)
    if certificate.ray:
        raise ValueError("optimality certificate cannot contain a recession ray")
    Q, c, G, h, A, b = canonical(problem)
    n = len(problem.c)
    if len(certificate.primal) != n or len(certificate.inequality_dual) != G.rows or len(certificate.equality_dual) != A.rows:
        raise ValueError("certificate dimensions do not match canonical constraints (including bounds)")
    x = sp.Matrix(certificate.primal)
    lam = sp.Matrix(G.rows, 1, certificate.inequality_dual)
    nu = sp.Matrix(A.rows, 1, certificate.equality_dual)
    residual = Q*x+c+G.T*lam+A.T*nu
    slack = h-G*x
    psd, pivots = exact_psd(Q)
    checks = {
        "primal_inequalities": all(v.is_nonnegative is True for v in slack),
        "primal_equalities": all(sp.cancel(v) == 0 for v in A*x-b),
        "dual_nonnegative": all(v.is_nonnegative is True for v in lam),
        "stationarity": all(sp.cancel(v) == 0 for v in residual),
        "complementarity": all(sp.cancel(lam[i]*slack[i]) == 0 for i in range(G.rows)),
        "convexity": psd is True,
        "integrality": all(not required or x[i].is_integer is True for i, required in enumerate(problem.integrality)),
    }
    primal = (x.T*Q*x)[0]/2+(c.T*x)[0]
    # At stationary x this equals the Lagrangian dual value, also for singular Q.
    dual = -(x.T*Q*x)[0]/2-(h.T*lam)[0]-(b.T*nu)[0]
    checks["zero_duality_gap"] = sp.cancel(primal-dual) == 0
    accepted = all(checks.values())
    exact_data = all(v.is_Rational for m in (Q, c, G, h, A, b) for v in m)
    exact_witness = all(v.is_Rational for v in (*certificate.primal, *certificate.inequality_dual, *certificate.equality_dual))
    trust = cap_trust(_problem_trust(problem), certificate_trust, arithmetic_trust((*certificate.primal, *certificate.inequality_dual, *certificate.equality_dual)))
    # Decimal residual zero is a floating equality, not an exact certificate.
    certified = accepted and exact_data and exact_witness and trust == "exact"
    sign = 1 if problem.sense == "min" else -1
    bundle = EvidenceBundle(computation=[ComputationEvidence(engine="optimization_certificate_checker",
        method="original_data_kkt", arithmetic=trust, trust=trust)])
    bundle.certificate.append(CertificateEvidence(certificate_type="convex_kkt",
        claim="global_optimum", witness=certificate.model_dump(mode="json"), verifier="original_data_kkt",
        verified=certified, trust="exact" if certified else "unknown",
        role="required" if certified else "diagnostic", metadata={"checks": checks, "psd_pivots": [str(v) for v in pivots]}))
    return EngineeringResult(operation="verify_certificate", status="verified" if certified else "candidate",
        trust=trust, value=sign*primal,
        details={"conclusion": "certified_global_optimum" if certified else "uncertified_candidate",
                 "certificate_checks": checks, "accepted": certified, "primal": certificate.primal,
                 "primal_value": sign*primal, "dual_value": sign*dual, "duality_gap": sp.cancel(primal-dual),
                 "stationarity_residual": tuple(residual), "slack": tuple(slack)},
        verification={"certificate_accepted": True} if certified else {}, claim_evidence={"optimality": bundle})


def _rational_candidate(values, denominator):
    from fractions import Fraction
    import math
    out = []
    for value in values:
        if not math.isfinite(float(value)):
            raise ValueError("solver returned nonfinite witness")
        f = Fraction(float(value)).limit_denominator(denominator)
        out.append(sp.Rational(f.numerator, f.denominator))
    return tuple(out)


def _numeric_report(problem, x, lam, nu, solver_status, message, method, tolerance):
    import numpy as np
    Q, c, G, h, A, b = canonical(problem)
    def arr(m):
        return np.asarray(m.tolist(), dtype=float).reshape(m.shape)
    q, cv, g, hv, ae, bv = arr(Q), arr(c).ravel(), arr(G), arr(h).ravel(), arr(A), arr(b).ravel()
    if not all(np.all(np.isfinite(v)) for v in (x, lam, nu)):
        raise ValueError("solver returned nonfinite candidate or multipliers")
    slack = hv-g@x
    primal_res = max(0.0, float(np.max(-slack, initial=0)), float(np.max(abs(ae@x-bv), initial=0)))
    stationarity = q@x+cv+g.T@lam+ae.T@nu
    dual_res = max(float(np.max(abs(stationarity), initial=0)), float(np.max(-lam, initial=0)), 0.0)
    complementarity = float(np.max(abs(lam*slack), initial=0))
    primal = float(x@q@x/2+cv@x)
    dual = float(-x@q@x/2-hv@lam-bv@nu)
    integer_res = max([abs(float(x[i])-round(float(x[i]))) for i, flag in enumerate(problem.integrality) if flag] or [0.0])
    feasible = primal_res <= tolerance and integer_res <= tolerance
    trust = cap_trust(problem.input_trust, "numeric")
    bundle = EvidenceBundle(computation=[ComputationEvidence(engine=method, method=method,
        arithmetic="float64", precision=53, trust=trust)], numerical=[NumericalEvidence(
        precision=53, residual={"primal": primal_res, "dual": dual_res, "complementarity": complementarity,
                                "integrality": integer_res, "gap": primal-dual}, trust=trust)])
    return EngineeringResult(operation="solve", status="candidate", trust=trust,
        value=primal if problem.sense == "min" else -primal,
        details={"conclusion": "numerically_feasible_incumbent" if feasible else "candidate",
                 "primal": [float(v) for v in x], "solver_status": solver_status, "solver_message": message,
                 "residuals": {"primal": primal_res, "dual": dual_res, "complementarity": complementarity,
                               "integrality": integer_res}, "primal_value": primal if problem.sense == "min" else -primal,
                 "dual_value": dual if problem.sense == "min" else -dual,
                 "duality_gap": primal-dual, "tolerance": tolerance, "global_optimality_certified": False},
        claim_evidence={"candidate_computation": bundle})


def _isolated_scipy_optimize(family, q, cv, g, hv, ae, bv, integrality,
                             max_iterations, time_limit, tolerance):
    """Native candidate search only; all checking remains in the parent."""
    import numpy as np
    from scipy import optimize
    if family == "lp":
        raw = optimize.linprog(cv, A_ub=g if g.shape[0] else None,
            b_ub=hv if g.shape[0] else None, A_eq=ae if ae.shape[0] else None,
            b_eq=bv if ae.shape[0] else None, bounds=(None, None), method="highs",
            options={"maxiter": max_iterations, "time_limit": time_limit*.9})
        return {"x": raw.x, "status": int(raw.status), "message": str(raw.message),
                "inequality_dual": None if not g.shape[0] or raw.ineqlin.marginals is None else -raw.ineqlin.marginals,
                "equality_dual": None if not ae.shape[0] or raw.eqlin.marginals is None else -raw.eqlin.marginals}
    constraints = []
    if g.shape[0]: constraints.append(optimize.LinearConstraint(g, -np.inf, hv))
    if ae.shape[0]: constraints.append(optimize.LinearConstraint(ae, bv, bv))
    if family == "milp":
        raw = optimize.milp(cv, integrality=np.asarray(integrality),
            bounds=optimize.Bounds(-np.inf, np.inf), constraints=constraints,
            options={"node_limit": max_iterations, "time_limit": time_limit*.9})
        return {"x": raw.x, "status": int(raw.status), "message": str(raw.message),
                "mip_dual_bound": getattr(raw, "mip_dual_bound", None)}
    start = np.linalg.lstsq(ae, bv, rcond=None)[0] if ae.shape[0] else np.zeros(len(cv))
    raw = optimize.minimize(lambda x: float(x@q@x/2+cv@x), start,
        jac=lambda x: q@x+cv, method="SLSQP", constraints=constraints,
        options={"maxiter": max_iterations, "ftol": tolerance})
    return {"x": raw.x, "status": int(raw.status), "message": str(raw.message)}


def solve(problem, *, mode="exact", max_iterations=1000, tolerance=1e-9,
          time_limit=30.0, reconstruction_denominator=1_000_000):
    Q, c, G, h, A, b = canonical(problem)
    n = len(problem.c)
    if mode == "exact":
        if problem.family != "lp":
            raise NotImplementedError("exact search currently supports LP; QP/MILP use numeric search plus exact certificate verification")
        if not all(v.is_Rational for matrix in (Q, c, G, h, A, b) for v in matrix):
            raise ValueError("exact LP search requires rational coefficients; use mode='numeric' explicitly")
        from sympy.solvers.simplex import linprog, InfeasibleLPError, UnboundedLPError
        try:
            # Dual is an independent optimization; certificate substitution is the authority.
            # Explicit positive/negative splitting avoids relying on an engine's
            # free-variable bound transformation. Search still needs checking.
            gp = G.row_join(-G) if G.rows else sp.zeros(1, 2*n)
            ap = A.row_join(-A) if A.rows else None
            _, split = linprog(list(c)+list(-c), gp, h if G.rows else sp.zeros(1, 1),
                               ap, b if A.rows else None)
            primal = [split[i]-split[n+i] for i in range(n)]
            dual_c = list(h)+list(b)+list(-b)
            stationarity = G.T.row_join(A.T).row_join(-A.T)
            if not dual_c:
                lam, nu = (), ()
            else:
                _, dual = linprog(dual_c, sp.zeros(1, len(dual_c)), sp.zeros(1, 1),
                                  A_eq=stationarity, b_eq=-c)
                lam = tuple(dual[:G.rows])
                nu = tuple(dual[G.rows+i]-dual[G.rows+A.rows+i] for i in range(A.rows))
        except (InfeasibleLPError, UnboundedLPError) as exc:
            from .optimization_proofs import find_outcome
            # A failure can arise in the primal or dual search, whose outcome
            # labels are not interchangeable. Try independently checked witnesses.
            proof = find_outcome(problem, "infeasible")
            if proof is None:
                proof = find_outcome(problem, "unbounded")
            if proof is not None:
                proof.operation = "solve"
                return proof
            # Failure to find a witness never upgrades the solver report.
            return EngineeringResult(operation="solve", status="unknown", trust="unknown",
                details={"solver_status": type(exc).__name__, "conclusion": "unverified_solver_report"},
                diagnostics=["No independently checked infeasibility/unboundedness certificate was returned."])
        cert = OptimizationCertificate(primal=tuple(sp.Rational(v) for v in primal),
            inequality_dual=tuple(sp.Rational(v) for v in lam), equality_dual=tuple(sp.Rational(v) for v in nu))
        result = verify_certificate(problem, cert)
        if not result.details.get("accepted"):
            from .optimization_proofs import find_outcome
            outcome = find_outcome(problem, "infeasible")
            if outcome is None:
                outcome = find_outcome(problem, "unbounded")
            if outcome is not None:
                outcome.operation = "solve"
                return outcome
        result.operation = "solve"
        result.details.update({"certificate": cert.model_dump(mode="json"),
                               "certificate_model": cert,
                               "solver": "sympy_exact_simplex"})
        return result
    if mode != "numeric":
        raise ValueError("mode must be exact or numeric")
    import numpy as np
    from .engines import run_in_subprocess
    def arr(matrix):
        # Reject overflow during conversion, rather than letting a solver misclassify it.
        return numeric_array(tuple(matrix), real=True).reshape(matrix.shape)
    q, cv, g, hv, ae, bv = arr(Q), arr(c).ravel(), arr(G), arr(h).ravel(), arr(A), arr(b).ravel()
    raw = run_in_subprocess(_isolated_scipy_optimize, time_limit, problem.family,
        q, cv, g, hv, ae, bv, problem.integrality, max_iterations, time_limit, tolerance)
    if problem.family == "lp":
        if raw["x"] is None:
            from .optimization_proofs import find_outcome
            proof = find_outcome(problem, "infeasible" if raw["status"] == 2 else "unbounded" if raw["status"] == 3 else "unknown")
            if proof is not None:
                proof.operation = "solve"
                return proof
            return EngineeringResult(operation="solve", status="unknown", trust="unknown",
                details={"solver_status": raw["status"], "solver_message": raw["message"],
                         "conclusion": "unverified_solver_report",
                         "process_isolation": "fresh_interpreter_hard_killable"})
        x = raw["x"]
        lam = raw["inequality_dual"] if raw["inequality_dual"] is not None else np.zeros(G.rows)
        nu = raw["equality_dual"] if raw["equality_dual"] is not None else np.zeros(A.rows)
        method = "scipy_highs_lp"
    elif problem.family == "milp":
        if raw["x"] is None:
            from .optimization_proofs import find_outcome
            proof = find_outcome(problem, "infeasible" if raw["status"] == 2 else "unbounded" if raw["status"] == 3 else "unknown")
            if proof is not None:
                proof.operation = "solve"
                return proof
            return EngineeringResult(operation="solve", status="unknown", trust="unknown",
                details={"solver_status": raw["status"], "solver_message": raw["message"],
                         "conclusion": "unverified_solver_report",
                         "process_isolation": "fresh_interpreter_hard_killable"})
        x, lam, nu, method = raw["x"], np.zeros(G.rows), np.zeros(A.rows), "scipy_highs_milp"
    else:
        x = raw["x"]
        active = np.flatnonzero(abs(g@x-hv) <= tolerance*10)
        # This least-squares multiplier estimate is only a witness candidate.
        basis = np.column_stack((g[active].T, ae.T))
        multipliers = np.linalg.lstsq(basis, -(q@x+cv), rcond=None)[0]
        lam, nu = np.zeros(G.rows), multipliers[len(active):]
        lam[active] = multipliers[:len(active)]
        method = "scipy_slsqp_qp"
    report = _numeric_report(problem, x, lam, nu, raw["status"], raw["message"], method, tolerance)
    report.details["process_isolation"] = "fresh_interpreter_hard_killable"
    if problem.family != "milp":
        cert = OptimizationCertificate(primal=_rational_candidate(x, reconstruction_denominator),
            inequality_dual=_rational_candidate(lam, reconstruction_denominator),
            equality_dual=_rational_candidate(nu, reconstruction_denominator))
        verified = verify_certificate(problem, cert)
        if verified.details["accepted"]:
            verified.operation = "solve"
            verified.details.update({"certificate": cert.model_dump(mode="json"),
                                     "certificate_model": cert, "solver": method,
                                     "search_diagnostics": report.details,
                                     "reconstruction_max_denominator": reconstruction_denominator})
            # Search is diagnostic: the certificate proves the exact original problem.
            diagnostic = report.claim_evidence["candidate_computation"].model_copy(deep=True)
            for group in (diagnostic.computation, diagnostic.numerical):
                for item in group:
                    item.role = "diagnostic"
            verified.claim_evidence["optimality"].computation.extend(diagnostic.computation)
            verified.claim_evidence["optimality"].numerical.extend(diagnostic.numerical)
            return verified
        report.details.update({"candidate_certificate": cert.model_dump(mode="json"),
                               "candidate_certificate_model": cert,
                               "reconstruction_max_denominator": reconstruction_denominator})
    if problem.family == "milp":
        report.details["solver_bound"] = float(raw["mip_dual_bound"]) if raw.get("mip_dual_bound") is not None else None
        if problem.sense == "max" and report.details["solver_bound"] is not None:
            report.details["solver_bound"] *= -1
        report.details["solver_bound_verified"] = False
    return report
