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
        kinds: set[str] = set()
        for expression in [left, right, *assumptions]:
            _scan_ir(expression, set(), set(), kinds)
        approximate = "real" in kinds
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
            trust=(TrustLevel.NUMERIC if approximate else TrustLevel.SYMBOLIC) if yes else TrustLevel.UNKNOWN,
            detail={"difference": str(diff)}))
        detail["symbolic_difference"] = str(diff)

        # A closed rational counterexample needs no SMT solver. Do not apply
        # this shortcut to variables, approximate ancestry or conditional goals.
        if not approximate and not assumptions and not names and not yes:
            lvalue, rvalue = sym.to_sympy(left, env), sym.to_sympy(right, env)
            if lvalue.is_Rational and rvalue.is_Rational and lvalue != rvalue:
                witness = {"left_value": str(lvalue), "right_value": str(rvalue)}
                detail["counterexample"] = witness
                evidence.append(EngineEvidence(engine="rational", capability="exact_rational_inequality",
                    status=VerificationStatus.DISPROVED, trust=TrustLevel.EXACT,
                    detail={"counterexample": witness, "arithmetic": "exact rational evaluation"}))

        z3: Z3Engine = self.router.engine("z3")  # type: ignore[assignment]
        if approximate:
            evidence.append(EngineEvidence(engine="z3", capability="counterexample",
                status=VerificationStatus.UNKNOWN, role="diagnostic",
                detail={"unsupported_fragment": True},
                error="Exact SMT is disabled for approximate decimal inputs or assumptions"))
        elif z3.available:
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
        refuted = any(e.status == VerificationStatus.DISPROVED for e in evidence)
        if formal and not approximate and not refuted:
            try:
                ls, script, tactic, error = lean.prove_equivalence(left, right, assumptions)
                capability=f"formal_{tactic}"
                if ls == "proved":
                    detail["lean_certificate"] = script
                    detail["lean_tactic"] = tactic
                    evidence.append(EngineEvidence(engine="lean", capability=capability,
                        status=VerificationStatus.PROVED, trust=TrustLevel.FORMAL,
                        detail={"tactic": tactic, "certificate": script}))
                elif ls == "unavailable":
                    evidence.append(EngineEvidence(engine="lean", capability=capability,
                        status="unavailable", trust=TrustLevel.UNKNOWN, error=error,
                        detail={"candidate_generated": True, "checked": False, "tactic": tactic}))
                else:
                    evidence.append(EngineEvidence(engine="lean", capability=capability,
                        status="error", trust=TrustLevel.UNKNOWN, error=error,
                        detail={"candidate_generated": True, "checked": False, "tactic": tactic}))
                if ls != "proved":
                    detail["lean_candidate"] = {"script": script, "tactic": tactic,
                                                "checked": False, "status": ls}
            except (TypeError, ValueError) as exc:
                evidence.append(EngineEvidence(engine="lean", capability="formal_certificate",
                    status=VerificationStatus.UNKNOWN, trust=TrustLevel.UNKNOWN,
                    detail={"unsupported_fragment": True}, error=str(exc)))

        # A declined attempt is diagnostic. Successful independent verifiers
        # support alternative paths; only evidence for the selected conclusion
        # may establish it (a proof of P cannot strengthen a refutation of P).
        conclusion = VerificationStatus.DISPROVED if refuted else VerificationStatus.PROVED
        for record in evidence:
            record.role = "required" if record.status == conclusion else "diagnostic"
            record.support_path = f"verifier:{record.engine}"
        if refuted:
            return VerificationStatus.DISPROVED, TrustLevel.EXACT, evidence, detail
        if any(e.engine == "lean" and e.status == VerificationStatus.PROVED for e in evidence):
            return VerificationStatus.PROVED, TrustLevel.FORMAL, evidence, detail
        if any(e.engine == "z3" and e.status == VerificationStatus.PROVED for e in evidence):
            return VerificationStatus.PROVED, TrustLevel.EXACT, evidence, detail
        if any(e.engine == "sympy" and e.status == VerificationStatus.PROVED for e in evidence):
            return VerificationStatus.PROVED, TrustLevel.NUMERIC if approximate else TrustLevel.SYMBOLIC, evidence, detail
        return VerificationStatus.UNKNOWN, TrustLevel.UNKNOWN, evidence, detail
