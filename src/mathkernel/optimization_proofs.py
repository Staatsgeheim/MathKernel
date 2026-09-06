# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Small independent checkers for Farkas, recession-ray and MILP tree proofs.

Search may fail. Only original-data substitution and a complete partition tree
establish mathematical outcomes. No solver flags enter these checkers.
"""
from __future__ import annotations
import heapq
import time
from typing import Literal
import sympy as sp
from pydantic import StrictInt, model_validator
from .engineering import EngineeringModel, EngineeringResult, arithmetic_trust, cap_trust, checked_result
from .optimization import OptimizationProblem, OptimizationCertificate, canonical, _problem_trust


def _is_exact(problem, entries):
    return (_problem_trust(problem) == "exact" and
            all(x.is_Rational for m in canonical(problem) for x in m) and
            all(x.is_Rational for x in entries))


def verify_outcome(problem, certificate, *, certificate_trust="exact"):
    Q, c, G, h, A, b = canonical(problem)
    n = len(problem.c)
    trust = cap_trust(_problem_trust(problem), certificate_trust, arithmetic_trust(
        (*certificate.primal, *certificate.inequality_dual, *certificate.equality_dual, *certificate.ray)))
    if certificate.kind == "infeasible":
        if certificate.primal or certificate.ray or len(certificate.inequality_dual) != G.rows or len(certificate.equality_dual) != A.rows:
            raise ValueError("Farkas witness needs exactly the canonical inequality/equality multipliers")
        lam = sp.Matrix(G.rows, 1, certificate.inequality_dual)
        nu = sp.Matrix(A.rows, 1, certificate.equality_dual)
        contradiction = (h.T*lam)[0]+(b.T*nu)[0]
        checks = {"nonnegative_multipliers": all(x.is_nonnegative is True for x in lam),
                  "zero_combined_coefficients": all(x == 0 for x in G.T*lam+A.T*nu),
                  "strict_contradiction": contradiction.is_negative is True}
        details = {"contradiction": contradiction}
    elif certificate.kind == "unbounded":
        if len(certificate.primal) != n or len(certificate.ray) != n or certificate.inequality_dual or certificate.equality_dual:
            raise ValueError("unboundedness witness needs a feasible anchor and recession ray")
        x, d = sp.Matrix(certificate.primal), sp.Matrix(certificate.ray)
        derivative = (c.T*d)[0]
        checks = {"linear_objective": all(v == 0 for v in Q),
                  "anchor_inequalities": all(v.is_nonnegative is True for v in h-G*x),
                  "anchor_equalities": all(v == 0 for v in A*x-b),
                  "recession_inequalities": all(v.is_nonpositive is True for v in G*d),
                  "recession_equalities": all(v == 0 for v in A*d),
                  "strict_improvement": derivative.is_negative is True,
                  "integer_anchor_and_step": all(not flag or
                      (x[i].is_integer is True and d[i].is_integer is True)
                      for i, flag in enumerate(problem.integrality))}
        details = {"canonical_directional_objective": derivative,
                   "ray_parameter": "nonnegative integers" if any(problem.integrality) else "nonnegative reals"}
    else:
        raise ValueError("outcome certificate must be infeasible or unbounded")
    entries = (*certificate.primal, *certificate.ray, *certificate.inequality_dual, *certificate.equality_dual)
    accepted = all(checks.values()) and _is_exact(problem, entries) and certificate_trust == "exact"
    result = checked_result("verify_certificate", None, method=f"exact_{certificate.kind}_witness",
        trust=trust, checks={"certificate_accepted": True} if accepted else {},
        witness=certificate.model_dump(mode="json"), candidate=not accepted,
        details={**details, "accepted": accepted, "certificate_checks": checks,
                 "conclusion": f"certified_{certificate.kind}" if accepted else "uncertified_candidate",
                 "certificate": certificate.model_dump(mode="json")})
    if accepted:
        result.status = certificate.kind
    return result


def _feasible_point(G, h, A, b, n):
    """Candidate search only. Explicit splitting avoids free-bound ambiguity."""
    from sympy.solvers.simplex import linprog
    _, v = linprog([0]*(2*n), G.row_join(-G) if G.rows else sp.zeros(1,2*n),
                   h if G.rows else sp.zeros(1,1), A.row_join(-A) if A.rows else None,
                   b if A.rows else None)
    return tuple(sp.Rational(v[i]-v[n+i]) for i in range(n))


def find_outcome(problem, kind):
    """Find a rational witness, then hand it to the independent checker."""
    from sympy.solvers.simplex import linprog, InfeasibleLPError, UnboundedLPError
    if _problem_trust(problem) != "exact" or not _is_exact(problem, ()):
        return None
    Q,c,G,h,A,b = canonical(problem)
    n=len(problem.c)
    try:
        if kind == "infeasible":
            width=G.rows+2*A.rows
            if width == 0:
                return None
            equalities=G.T.row_join(A.T).row_join(-A.T)
            equalities=equalities.col_join(h.T.row_join(b.T).row_join(-b.T))
            target=sp.zeros(n,1).col_join(sp.Matrix([-1]))
            _, values=linprog([0]*width,sp.zeros(1,width),sp.zeros(1,1),equalities,target)
            cert=OptimizationCertificate(kind="infeasible",
                inequality_dual=tuple(sp.Rational(x) for x in values[:G.rows]),
                equality_dual=tuple(sp.Rational(values[G.rows+i]-values[G.rows+A.rows+i]) for i in range(A.rows)))
        elif kind == "unbounded" and not problem.Q:
            anchor=_feasible_point(G,h,A,b,n)
            ray=_feasible_point(G,sp.zeros(G.rows,1),A.col_join(c.T),
                                sp.zeros(A.rows,1).col_join(sp.Matrix([-1])),n)
            integer_denominators=[ray[i].q for i,flag in enumerate(problem.integrality) if flag]
            scale=sp.ilcm(1,*integer_denominators) if integer_denominators else 1
            cert=OptimizationCertificate(kind="unbounded",primal=anchor,ray=tuple(scale*v for v in ray))
        else:
            return None
        result=verify_outcome(problem,cert)
        return result if result.details["accepted"] else None
    except (InfeasibleLPError, UnboundedLPError, ValueError, ZeroDivisionError):
        return None


class MILPProofNode(EngineeringModel):
    kind: Literal["split", "bound", "infeasible", "open"]
    variable: StrictInt | None = None
    split: sp.Expr | None = None
    left: StrictInt | None = None
    right: StrictInt | None = None
    certificate: OptimizationCertificate | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.kind == "split":
            if self.variable is None or self.variable < 0 or self.split is None or self.split.is_Integer is not True:
                raise ValueError("split requires a variable index and an exact integer threshold")
            if self.left is None or self.right is None or self.left < 0 or self.right < 0:
                raise ValueError("split requires two child indices")
            if self.certificate is not None:
                raise ValueError("split nodes do not carry leaf certificates")
        elif any(v is not None for v in (self.variable,self.split,self.left,self.right)):
            raise ValueError("leaf cannot carry branch fields")
        elif self.kind in {"bound","infeasible"} and self.certificate is None:
            raise ValueError("closed leaf requires a certificate")
        return self


class MILPCertificate(EngineeringModel):
    incumbent: tuple[sp.Expr, ...] = ()
    nodes: tuple[MILPProofNode, ...]

    @model_validator(mode="after")
    def validate_proof(self):
        from .engineering import validate_scalars
        validate_scalars(self.incumbent,real=True)
        if not self.nodes:
            raise ValueError("MILP certificate needs a root node")
        return self


def _branch(problem, variable, threshold, upper):
    row=[sp.S.Zero]*len(problem.c)
    row[variable]=sp.S.One if upper else -sp.S.One
    return problem.model_copy(update={"A_ub": (*problem.A_ub,tuple(row)),
        "b_ub": (*problem.b_ub,threshold if upper else -threshold)})


def _feasible_integer(problem, point):
    if len(point) != len(problem.c) or not all(v.is_Rational for v in point):
        return False
    _,_,G,h,A,b=canonical(problem)
    x=sp.Matrix(point)
    return (all(v.is_nonnegative is True for v in h-G*x) and all(v == 0 for v in A*x-b) and
            all(not flag or x[i].is_integer is True for i,flag in enumerate(problem.integrality)))


def verify_milp(problem, proof, *, max_nodes=255, certificate_trust="exact"):
    """Verify coverage and original-data leaf bounds; no optimization is run."""
    from .optimization import verify_certificate
    if problem.family != "milp":
        raise ValueError("MILP tree certificates require a linear mixed-integer problem")
    if len(proof.nodes)>max_nodes:
        raise ValueError("MILP proof exceeds max_milp_nodes")
    exact=_is_exact(problem,proof.incumbent) and certificate_trust == "exact"
    feasible=bool(proof.incumbent) and _feasible_integer(problem,proof.incumbent)
    _,c,_,_,_,_=canonical(problem)
    incumbent_value=(c.T*sp.Matrix(proof.incumbent))[0] if feasible else None
    seen=set();stack=[(0,problem)]; valid=exact and (feasible or not proof.incumbent)
    failures=[];closed=0
    while stack:
        index,subproblem=stack.pop()
        if index in seen or not 0<=index<len(proof.nodes):
            valid=False;failures.append("reused, cyclic or out-of-range child");continue
        seen.add(index);node=proof.nodes[index]
        if node.kind == "split":
            if node.variable >= len(problem.c) or not problem.integrality[node.variable]:
                valid=False;failures.append("split variable is not integral");continue
            stack.extend([(node.right,_branch(subproblem,node.variable,node.split+1,False)),
                          (node.left,_branch(subproblem,node.variable,node.split,True))])
        elif node.kind == "open":
            valid=False;failures.append("open branch")
        else:
            relaxation=subproblem.model_copy(update={"integrality":()})
            try:
                result=verify_certificate(relaxation,node.certificate)
                accepted=result.details.get("accepted",False)
                if node.kind == "infeasible":
                    accepted=accepted and node.certificate.kind=="infeasible"
                else:
                    # This lower bound is certified on the full node relaxation.
                    accepted=accepted and node.certificate.kind=="optimal" and feasible
                    if accepted:
                        bound=sp.Matrix(node.certificate.primal).dot(c)
                        accepted=(bound-incumbent_value).is_nonnegative is True
                if not accepted:
                    valid=False;failures.append(f"invalid {node.kind} leaf {index}")
                else:
                    closed+=1
            except (ValueError,TypeError):
                valid=False;failures.append(f"invalid certificate dimensions at leaf {index}")
    if len(seen)!=len(proof.nodes):
        valid=False;failures.append("unreachable nodes")
    conclusion="certified_global_optimum" if valid and feasible else "certified_infeasible" if valid else "incomplete_or_invalid_proof"
    trust=cap_trust(_problem_trust(problem),certificate_trust,arithmetic_trust(proof.incumbent))
    result=checked_result("verify_milp_certificate", None if not feasible else
        (incumbent_value if problem.sense=="min" else -incumbent_value),
        method="exact_integer_partition_tree",trust=trust,checks={"certificate_accepted":True} if valid else {},
        witness=proof.model_dump(mode="json"),candidate=not valid,
        details={"accepted":valid,"conclusion":conclusion,"incumbent":proof.incumbent,
                 "incumbent_feasible":feasible,"closed_leaves":closed,"nodes":len(proof.nodes),"failures":failures})
    if valid and not feasible:
        result.status="infeasible"
    return result


def solve_milp(problem, *, mode="exact", max_nodes=255, time_limit=30.0):
    """Deterministic best-bound branching with independently certified LP leaves."""
    from .optimization import solve, verify_certificate
    if problem.family!="milp" or not _is_exact(problem,()):
        raise ValueError("certified MILP search requires rational exact linear problem data")
    deadline=time.monotonic()+time_limit
    nodes=[MILPProofNode(kind="open")]
    queue=[(sp.S.NegativeInfinity,0,problem)]
    incumbent=();best=None;solved=0;reason=None
    # Each child owns an immutable model; canonical matrices are built on demand.
    while queue:
        if time.monotonic()>=deadline:
            reason="time budget";break
        _,index,node_problem=heapq.heappop(queue)
        relaxation=node_problem.model_copy(update={"integrality":()})
        result=solve(relaxation,mode=mode,time_limit=max(.001,deadline-time.monotonic()))
        solved+=1
        if result.status=="infeasible" and result.details.get("accepted"):
            cert=_certificate_from_result(result)
            nodes[index]=MILPProofNode(kind="infeasible",certificate=cert);continue
        if result.details.get("conclusion")!="certified_global_optimum":
            reason="LP relaxation lacks an accepted finite optimum certificate";break
        cert=_certificate_from_result(result)
        bound=sp.Matrix(cert.primal).dot(canonical(relaxation)[1])
        if best is not None and bound>=best:
            nodes[index]=MILPProofNode(kind="bound",certificate=cert);continue
        fractional=next((i for i,flag in enumerate(problem.integrality) if flag and cert.primal[i].is_integer is not True),None)
        if fractional is None:
            incumbent=cert.primal;best=bound
            nodes[index]=MILPProofNode(kind="bound",certificate=cert);continue
        if len(nodes)+2>max_nodes:
            reason="node budget";break
        threshold=sp.floor(cert.primal[fractional])
        left,right=len(nodes),len(nodes)+1
        nodes[index]=MILPProofNode(kind="split",variable=fractional,split=threshold,left=left,right=right)
        nodes.extend([MILPProofNode(kind="open"),MILPProofNode(kind="open")])
        heapq.heappush(queue,(bound,left,_branch(node_problem,fractional,threshold,True)))
        heapq.heappush(queue,(bound,right,_branch(node_problem,fractional,threshold+1,False)))
    proof=MILPCertificate(incumbent=incumbent,nodes=tuple(nodes))
    result=verify_milp(problem,proof,max_nodes=max_nodes)
    result.operation="certify_milp"
    result.details.update({"certificate":proof.model_dump(mode="json"),"lp_relaxations":solved,
                           "termination":reason or "complete", "search_mode":mode})
    if reason:
        result.diagnostics.append(f"Search stopped at {reason}; open branches prevent an optimum/infeasibility claim.")
    return result,proof


def _certificate_from_result(result):
    data=result.details["certificate"]
    # Already produced inside the trusted search path. No string parser is used.
    def numbers(key):
        return tuple(sp.Rational(v) for v in data.get(key,()))
    return OptimizationCertificate(kind=data.get("kind","optimal"),primal=numbers("primal"),
        inequality_dual=numbers("inequality_dual"),equality_dual=numbers("equality_dual"),ray=numbers("ray"))
