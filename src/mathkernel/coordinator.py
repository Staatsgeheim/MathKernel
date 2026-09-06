# =============================================================================
# MathKernel - coordinator
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
from .capabilities import CapabilityRouter
from .engines import LeanEngine, SymPyEngine, Z3Engine, symbol_env
from .models import EngineEvidence, Expr, MathContext, TrustLevel, VerificationStatus
from .reasoning import _walk as _scan_ir


class VerificationCoordinator:
    """Collect independent evidence and reconcile it conservatively."""

    def __init__(self, router: CapabilityRouter): self.router = router

    @staticmethod
    def _assumptions(context: MathContext | None) -> tuple[list[Expr], dict[str,str]]:
        if not context: return [], {}
        return [a.expression for a in context.assumptions], context.domains

    def equivalence(self, left: Expr, right: Expr, context: MathContext | None = None,
                    formal: bool = True) -> tuple[VerificationStatus, TrustLevel, list[EngineEvidence], dict]:
        assumptions, domains = self._assumptions(context)
        evidence: list[EngineEvidence] = []
        detail: dict = {}

        names: set[str] = set()
        _scan_ir(left, names, set(), set())
        _scan_ir(right, names, set(), set())
        env = symbol_env(sorted(names), domains or None,
                         context.symbol_properties if context else None)

        sym: SymPyEngine = self.router.engine("sympy")  # type: ignore[assignment]
        yes, diff = sym.prove_equivalence(left, right, env)
        evidence.append(EngineEvidence(engine="sympy", capability="symbolic_equivalence",
            status=VerificationStatus.PROVED if yes else VerificationStatus.UNKNOWN,
            trust=TrustLevel.SYMBOLIC, detail={"difference": str(diff)}))
        detail["symbolic_difference"] = str(diff)

        z3: Z3Engine = self.router.engine("z3")  # type: ignore[assignment]
        if z3.available:
            try:
                status, model = z3.counterexample_equivalence(left, right, assumptions, domains)
                zs = VerificationStatus(status)
                evidence.append(EngineEvidence(engine="z3", capability="counterexample", status=zs,
                    trust=TrustLevel.EXACT, detail={"counterexample": model} if model else {}))
                if model: detail["counterexample"] = model
            except (TypeError, ValueError) as exc:
                evidence.append(EngineEvidence(engine="z3", capability="counterexample",
                    status=VerificationStatus.UNKNOWN, trust=TrustLevel.UNKNOWN,
                    detail={"unsupported_fragment": True}, error=str(exc)))
        else:
            evidence.append(EngineEvidence(engine="z3", capability="counterexample", status="unavailable",
                trust=TrustLevel.UNKNOWN, error="z3-solver is not installed"))

        lean: LeanEngine = self.router.engine("lean")  # type: ignore[assignment]
        if formal:
            try:
                ls, script, tactic, error = lean.prove_equivalence(left, right, assumptions)
                detail["lean_certificate"] = script
                detail["lean_tactic"] = tactic
                capability=f"formal_{tactic}"
                if ls == "proved":
                    evidence.append(EngineEvidence(engine="lean", capability=capability,
                        status=VerificationStatus.PROVED, trust=TrustLevel.FORMAL, detail={"tactic": tactic}))
                elif ls == "unavailable":
                    evidence.append(EngineEvidence(engine="lean", capability=capability,
                        status="unavailable", trust=TrustLevel.UNKNOWN, error=error,
                        detail={"certificate_generated": True, "tactic": tactic}))
                else:
                    evidence.append(EngineEvidence(engine="lean", capability=capability,
                        status="error", trust=TrustLevel.UNKNOWN, error=error,
                        detail={"certificate_generated": True, "tactic": tactic}))
            except (TypeError, ValueError) as exc:
                evidence.append(EngineEvidence(engine="lean", capability="formal_certificate",
                    status=VerificationStatus.UNKNOWN, trust=TrustLevel.UNKNOWN,
                    detail={"unsupported_fragment": True}, error=str(exc)))

        if any(e.status == VerificationStatus.DISPROVED for e in evidence):
            return VerificationStatus.DISPROVED, TrustLevel.EXACT, evidence, detail
        if any(e.engine == "lean" and e.status == VerificationStatus.PROVED for e in evidence):
            return VerificationStatus.PROVED, TrustLevel.FORMAL, evidence, detail
        if any(e.engine == "z3" and e.status == VerificationStatus.PROVED for e in evidence):
            return VerificationStatus.PROVED, TrustLevel.EXACT, evidence, detail
        if any(e.engine == "sympy" and e.status == VerificationStatus.PROVED for e in evidence):
            return VerificationStatus.PROVED, TrustLevel.SYMBOLIC, evidence, detail
        return VerificationStatus.UNKNOWN, TrustLevel.UNKNOWN, evidence, detail
