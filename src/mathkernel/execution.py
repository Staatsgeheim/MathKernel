# =============================================================================
# MathKernel - execution
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations
import uuid
from typing import Any
import sympy as sp
from mathkernel_artifacts import (
    ComputationEvidence,
    EvidenceBundle,
    merge_evidence_bundles,
    tag_evidence_bundle,
)

from .models import (
    DerivationStep, EngineEvidence, MathContext, MathResult, Obligation,
    ObligationExecution, PlanExecution, ProblemPlan, RelationNode, ResultStatus,
    TrustLevel, VerificationStatus,
)
from .parallel import resolve_workers, thread_map
from .parser import parse_math
from .rendering import render_expr
from .solution import serialize_solution_set


_TRUST_RANK = {
    TrustLevel.UNKNOWN: 0,
    TrustLevel.HEURISTIC: 1,
    TrustLevel.EMPIRICAL: 2,
    TrustLevel.NUMERIC: 3,
    TrustLevel.NUMERIC_HIGH_PRECISION: 4,
    TrustLevel.INTERVAL_CERTIFIED: 5,
    TrustLevel.SYMBOLIC: 6,
    TrustLevel.EXACT: 7,
    TrustLevel.FORMAL: 8,
}


def _eid() -> str:
    return "exec_" + uuid.uuid4().hex[:16]


def _assumption_text(kernel: Any, context: MathContext | None) -> list[str]:
    if not context:
        return []
    return [render_expr(a.expression) for a in context.assumptions]


def reconcile_execution(plan: ProblemPlan, nodes: list[ObligationExecution]) -> tuple[str, str, TrustLevel, list[dict], dict]:
    """Conservatively reconcile obligation outcomes.

    Trust is attached to the *whole result*, not the strongest isolated sub-proof.
    A formally verified candidate does not make a solution set formally complete if
    completeness was only established symbolically.
    """
    by_action = {n.action: n for n in nodes}
    conflicts: list[dict] = []

    verify = by_action.get("verify_solution_candidates")
    complete = by_action.get("check_solution_completeness")
    domain = by_action.get("validate_context")

    if domain and domain.state == "refuted":
        conflicts.append({"type": "inconsistent_context", "obligation_id": domain.obligation_id})
    if verify and verify.state == "refuted":
        conflicts.append({"type": "invalid_candidate", "obligation_id": verify.obligation_id,
                          "detail": verify.result.get("invalid_candidates", [])})
    if complete and complete.state == "refuted":
        conflicts.append({"type": "missed_solution", "obligation_id": complete.obligation_id,
                          "detail": complete.result.get("counterexample")})

    failed = [n for n in nodes if n.state == "failed"]
    if conflicts:
        return "refuted", "refuted", TrustLevel.SYMBOLIC, conflicts, {
            "reason": "One or more independent checks contradicted the candidate result or context."
        }
    if failed and all(n.state in {"failed", "skipped"} for n in nodes):
        return "failed", "error", TrustLevel.UNKNOWN, conflicts, {"reason": "No executable obligation completed."}

    if plan.classification in {"algebraic_relation", "transcendental_equation"}:
        solve = by_action.get("solve_relation")
        if not solve or solve.state not in {"succeeded", "verified"}:
            return "partial", "unknown", TrustLevel.UNKNOWN, conflicts, {"reason": "No candidate solution set was produced."}
        if verify and verify.state == "verified":
            if complete and complete.state == "verified":
                final_trust = TrustLevel.EXACT
                completeness = "independently established"
            else:
                # SymPy is still the completeness authority when SMT/formal coverage is unavailable.
                final_trust = TrustLevel.SYMBOLIC
                completeness = "symbolic backend only"
            return "completed", "verified", final_trust, conflicts, {
                "candidate_soundness": "verified",
                "completeness": completeness,
                "solution_set": solve.result.get("solution_set"),
            }
        return "partial", "unknown", TrustLevel.SYMBOLIC, conflicts, {
            "candidate_soundness": verify.state if verify else "not_run",
            "solution_set": solve.result.get("solution_set"),
        }

    if plan.classification.endswith("relation"):
        sat = by_action.get("check_relation_satisfiability")
        if sat and sat.state == "verified":
            return "completed", "verified", sat.trust, conflicts, sat.result
        if sat and sat.state == "refuted":
            return "completed", "refuted", sat.trust, conflicts, sat.result
        return "partial", "unknown", TrustLevel.UNKNOWN, conflicts, sat.result if sat else {}

    norm = by_action.get("normalize_expression")
    if norm and norm.state == "succeeded":
        return "completed", "ok", norm.trust, conflicts, {"normalized_expression": norm.result.get("result")}
    return "partial", "unknown", TrustLevel.UNKNOWN, conflicts, {}


class ObligationExecutor:
    """Execute a ProblemPlan as a dependency-aware obligation DAG."""

    def __init__(self, kernel: Any):
        self.kernel = kernel

    @staticmethod
    def _typed_verification_state(result: MathResult) -> str:
        if result.status == "verified":
            return "verified"
        verification = result.data.get("verification")
        if isinstance(verification, dict) and verification:
            values = list(verification.values())
            if all(value is True for value in values):
                return "verified"
            if any(value is False for value in values):
                return "refuted"
        checks = result.data.get("verified_checks")
        if isinstance(checks, list) and checks:
            states = [
                check.get("status")
                for check in checks if isinstance(check, dict)
            ]
            if states and all(state == "verified" for state in states):
                return "verified"
        return "unknown"

    def execute_typed_apply(
        self, *, object_id: str, operation: str, parameters: dict,
        compute,
    ) -> MathResult:
        """Execute a typed operation as a live four-node obligation DAG."""
        try:
            record = self.kernel._get_math_object_record(object_id)
        except (TypeError, ValueError) as exc:
            return MathResult(
                ok=False, status="error", errors=[str(exc)],
                engine="math_object")
        if record is None:
            return MathResult(
                ok=False, status="error",
                errors=[f"Unknown object_id: {object_id}"],
                engine="obligation_executor")
        try:
            capability = self.kernel.router.registry.resolve(
                input_type=record["object_type"], operation=operation)
        except LookupError as exc:
            return MathResult(
                ok=False, status="error",
                semantic_status=ResultStatus.UNSUPPORTED,
                errors=[str(exc)], engine="capability_registry")

        try:
            dependency_records, parameter_trust = (
                self.kernel._typed_operation_dependencies(
                    object_id, capability, parameters)
            )
        except (ValueError, TypeError) as exc:
            return MathResult(
                ok=False, status="error", errors=[str(exc)],
                engine="mathir")
        ancestry_levels = [
            dep_record["input_trust"]
            for _, dep_record in dependency_records
        ]
        ancestry_levels.append(parameter_trust)
        ancestry_trust = min(
            ancestry_levels, key=lambda level: _TRUST_RANK[level])
        ancestry_bundle = EvidenceBundle(computation=[
            ComputationEvidence(
                engine="math_object",
                method="required_input",
                arithmetic="inherited",
                trust=dep_record["input_trust"].value,
                metadata={
                    "object_id": dep_id,
                    "object_type": dep_record["object_type"],
                    "sources": list(dep_record.get("sources", [])),
                },
            )
            for dep_id, dep_record in dependency_records
        ], justified_trust=ancestry_trust.value)
        if parameter_trust != TrustLevel.EXACT:
            ancestry_bundle.computation.append(ComputationEvidence(
                engine="mathir", method="operation_parameters",
                arithmetic="inherited", trust=parameter_trust.value,
                metadata={"operation": operation},
            ))

        ids = [self.kernel._id("obl") for _ in range(4)]
        plan_id = self.kernel._id("plan")
        plan = ProblemPlan(
            classification="typed_domain_operation",
            required_capabilities=[capability.name],
            strategy=[
                "validate_typed_object", "compute_candidate",
                "verify_domain_invariants", "reconcile_claim_evidence",
            ],
            obligations=[
                Obligation(
                    obligation_id=ids[0], kind="domain",
                    action="validate_typed_object",
                    statement=f"Validate stored {record['object_type']}",
                    produces=["validated_object"]),
                Obligation(
                    obligation_id=ids[1], kind="domain",
                    action="compute_candidate",
                    statement=f"Apply {operation} to {record['object_type']}",
                    required_capabilities=[capability.name],
                    preferred_engines=list(capability.engines),
                    depends_on=[ids[0]], produces=["candidate"],
                    capability_ref=capability.name,
                    resolved_handler=capability.handler,
                    parameters=dict(parameters)),
                Obligation(
                    obligation_id=ids[2], kind="verification",
                    action="verify_domain_invariants",
                    statement="Verify domain-specific identities and conditions",
                    depends_on=[ids[1]], produces=["verified_candidate"]),
                Obligation(
                    obligation_id=ids[3], kind="verification",
                    action="reconcile_claim_evidence",
                    statement="Reconcile claim-specific evidence conservatively",
                    depends_on=[ids[2]], produces=["reconciled_result"]),
            ],
            capability_routes={ids[1]: list(capability.engines)},
        )
        validation_bundle = ancestry_bundle.model_copy(deep=True)
        validation = ObligationExecution(
            obligation_id=ids[0], kind="domain",
            action="validate_typed_object", state="verified",
            engine="math_object", trust=record["input_trust"],
            result={"object_type": record["object_type"]},
            evidence_bundle=validation_bundle)
        self._record(plan.obligations[0], validation, object_id, [])

        result = compute()
        if result.ok:
            inherited_assumptions: list[str] = []
            for _, dep_record in dependency_records:
                for assumption in self.kernel._typed_record_assumptions(
                    dep_record
                ):
                    if assumption not in inherited_assumptions:
                        inherited_assumptions.append(assumption)
            for assumption in inherited_assumptions:
                if assumption not in result.assumptions_used:
                    result.assumptions_used.append(assumption)

            # Required input ancestry is part of every supported claim. This is
            # the central anti-laundering barrier for cross-object composition.
            if result.claim_evidence:
                result.claim_evidence = {
                    name: merge_evidence_bundles((
                        ancestry_bundle, bundle,
                    ))
                    for name, bundle in result.claim_evidence.items()
                }
            else:
                result.claim_evidence = {
                    "result": merge_evidence_bundles((
                        ancestry_bundle, result.evidence_bundle,
                    ))
                }
            result.evidence_bundle = merge_evidence_bundles((
                ancestry_bundle, result.evidence_bundle,
            ))
            if _TRUST_RANK[ancestry_trust] < _TRUST_RANK[result.trust]:
                result.trust = ancestry_trust
            result.populate_compatible_evidence()
            provenance = result.data.setdefault("provenance", {})
            transition = provenance.setdefault("arithmetic_transition", {})
            transition["required_input_trust"] = ancestry_trust.value
            transition["justified_output_trust"] = result.trust.value
            provenance["required_object_inputs"] = [
                {
                    "object_id": dep_id,
                    "object_type": dep_record["object_type"],
                    "trust": dep_record["input_trust"].value,
                }
                for dep_id, dep_record in dependency_records
            ]
            dependency_ids = [dep_id for dep_id, _ in dependency_records]
            for step in result.derivation:
                if _TRUST_RANK[result.trust] < _TRUST_RANK[step.trust]:
                    step.trust = result.trust
                for dep_id in dependency_ids:
                    if dep_id not in step.inputs:
                        step.inputs.append(dep_id)
                stored_step = self.kernel.derivations.get(step.step_id)
                if stored_step is not None:
                    stored_step.trust = step.trust
                    stored_step.inputs = list(step.inputs)

            # If the operation created a derived typed object, its future
            # ancestry must inherit the same ceiling. Never leave a stronger
            # trust level stored on a derived object than on its construction.
            derived_ids = list(dict.fromkeys([
                result.data.get("object_id"),
                *result.data.get("object_ids", {}).values(),
            ]))
            for derived_id in derived_ids:
                if derived_id and derived_id != object_id:
                    derived = self.kernel._get_math_object_record(str(derived_id))
                    if derived is not None and (
                        _TRUST_RANK[result.trust] <
                        _TRUST_RANK[derived["input_trust"]]
                    ):
                        derived["input_trust"] = result.trust
                        if hasattr(derived["value"], "input_trust"):
                            derived["value"] = derived["value"].model_copy(
                                update={"input_trust": result.trust.value})
                        derived["evidence_bundle"] = result.evidence_bundle.model_copy(deep=True)
                        derived["claim_evidence"] = {name: bundle.model_copy(deep=True)
                                                     for name, bundle in result.claim_evidence.items()}
                        self.kernel._persist_math_object(str(derived_id), derived)
                    if derived is not None:
                        derived["sources"] = list(dict.fromkeys([*derived["sources"], *dependency_ids]))
                        self.kernel._persist_math_object(str(derived_id), derived)

        compute_state = "succeeded" if result.ok else "failed"
        computed = ObligationExecution(
            obligation_id=ids[1], kind="domain",
            action="compute_candidate", state=compute_state,
            engine=result.engine, trust=result.trust,
            depends_on=[ids[0]], result={"value": result.data.get("value")},
            evidence_bundle=result.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in result.claim_evidence.items()
            },
            error="; ".join(result.errors) or None)
        self._record(plan.obligations[1], computed, object_id, [validation])

        verification_state = (
            self._typed_verification_state(result)
            if result.ok else "skipped"
        )
        if verification_state == "verified" and result.status != "verified":
            result.status = "verified"
            result.semantic_status = None
            result.populate_compatible_evidence()
        elif verification_state == "refuted" and result.status != "refuted":
            result.status = "refuted"
            result.semantic_status = None
            result.populate_compatible_evidence()
        verified = ObligationExecution(
            obligation_id=ids[2], kind="verification",
            action="verify_domain_invariants", state=verification_state,
            engine=result.engine, trust=result.trust,
            depends_on=[ids[1]],
            result={
                "verification": result.data.get("verification"),
                "checks": result.data.get("verified_checks", []),
            },
            evidence_bundle=result.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in result.claim_evidence.items()
            })
        self._record(plan.obligations[2], verified, object_id, [computed])

        reconciled = result.reconciled_trust()
        reconciliation_state = (
            "succeeded" if reconciled == result.trust else "failed")
        reconciled_run = ObligationExecution(
            obligation_id=ids[3], kind="verification",
            action="reconcile_claim_evidence", state=reconciliation_state,
            engine="evidence_reconciler", trust=reconciled,
            depends_on=[ids[2]],
            result={
                "declared_trust": result.trust.value,
                "reconciled_trust": reconciled.value,
            },
            evidence_bundle=result.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in result.claim_evidence.items()
            })
        self._record(
            plan.obligations[3], reconciled_run, object_id, [verified])

        runs = [validation, computed, verified, reconciled_run]
        execution = PlanExecution(
            execution_id=_eid(), plan_id=plan_id, expr_id=object_id,
            state="completed" if result.ok else "failed",
            final_status=(
                result.status if result.status in {
                    "verified", "refuted", "unknown", "error", "ok",
                } else "unknown"),
            final_trust=reconciled, obligations=runs,
            propagated_side_conditions=list(result.side_conditions),
            graph={
                "nodes": [
                    obligation.model_dump(mode="json")
                    for obligation in plan.obligations
                ],
                "edges": [
                    {"from": dependency, "to": obligation.obligation_id}
                    for obligation in plan.obligations
                    for dependency in obligation.depends_on
                ],
            },
            evidence_bundle=result.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in result.claim_evidence.items()
            })
        self.kernel.plans[plan_id] = plan
        self.kernel.plan_sources[plan_id] = (object_id, None)
        self.kernel.typed_plan_requests[plan_id] = {
            "object_id": object_id,
            "operation": operation,
            "parameters": dict(parameters),
        }
        self.kernel.executions[execution.execution_id] = execution
        self.kernel._persist_plan(plan_id)
        if self.kernel._store is not None:
            self.kernel._store.put_execution(
                execution.execution_id,
                execution.model_dump(mode="json"))

        result.data["plan"] = {
            "classification": plan.classification,
            "capability": capability.as_dict(),
            "obligations": [
                {
                    "action": run.action,
                    "state": run.state,
                    "obligation_id": run.obligation_id,
                }
                for run in runs
            ],
            "dag": plan.model_dump(mode="json"),
            "execution": execution.model_dump(mode="json"),
        }
        result.data["plan_id"] = plan_id
        result.data["execution_id"] = execution.execution_id
        recorded_steps = [
            self.kernel.derivations[run.derivation_step_id]
            for run in runs if run.derivation_step_id
        ]
        result.derivation.extend(recorded_steps)
        return result

    def _record(self, obligation: Obligation, run: ObligationExecution, expr_id: str,
                dep_runs: list[ObligationExecution]) -> None:
        run.populate_compatible_evidence()
        if run.evidence_bundle.is_empty() and run.state not in {
            "skipped", "failed", "pending", "running"
        }:
            run.evidence_bundle = EvidenceBundle(computation=[
                ComputationEvidence(
                    engine=run.engine or "unknown",
                    method=run.action,
                    arithmetic=run.trust.value,
                    trust=run.trust.value,
                )
            ])
            run.claim_evidence["result"] = run.evidence_bundle.model_copy(
                deep=True)
        parent_steps = [r.derivation_step_id for r in dep_runs if r.derivation_step_id]
        producer = self.kernel.expression_producers.get(expr_id)
        if producer and producer not in parent_steps:
            parent_steps.insert(0, producer)
        step = DerivationStep(
            step_id=self.kernel._id("step"),
            operation=f"obligation:{obligation.action}",
            inputs=[expr_id, obligation.obligation_id],
            parents=parent_steps,
            output=str(run.result)[:4000],
            conditions=[*run.assumptions_used, *run.side_conditions],
            engine=run.engine,
            trust=run.trust,
            evidence=run.evidence,
            evidence_bundle=run.evidence_bundle.model_copy(deep=True),
            claim_evidence={
                name: bundle.model_copy(deep=True)
                for name, bundle in run.claim_evidence.items()
            },
        )
        self.kernel._record(step)
        run.derivation_step_id = step.step_id

    @staticmethod
    def _blocked(dep_runs: list[ObligationExecution]) -> bool:
        return any(r.state in {"failed", "refuted", "skipped"} for r in dep_runs)

    def _validate_context(self, obligation: Obligation, context: MathContext | None, missing: list[str]) -> ObligationExecution:
        run = ObligationExecution(obligation_id=obligation.obligation_id, kind=obligation.kind, action=obligation.action,
                                  state="succeeded", engine="mathir", trust=TrustLevel.EXACT,
                                  depends_on=obligation.depends_on)
        run.assumptions_used = _assumption_text(self.kernel, context)
        run.side_conditions = list(missing)
        if context is None:
            run.result = {"consistency": "unchecked", "domains": {}, "missing_assumptions": missing}
            if missing:
                run.state = "unknown"; run.trust = TrustLevel.HEURISTIC
            return run
        if context.consistency == "inconsistent":
            run.state = "refuted"
            run.result = {"consistency": "inconsistent", "context_id": context.context_id}
            return run
        if self.kernel.z3.available and context.assumptions:
            try:
                c = self.kernel.z3.check_context([a.expression for a in context.assumptions], context.domains)
                context.consistency = c
                run.engine = "z3"
                run.result = {"consistency": c, "domains": context.domains, "missing_assumptions": missing}
                run.evidence.append(EngineEvidence(engine="z3", capability="smt",
                    status=(VerificationStatus.PROVED if c == "consistent" else VerificationStatus.DISPROVED if c == "inconsistent" else VerificationStatus.UNKNOWN),
                    trust=TrustLevel.EXACT, detail={"context_consistency": c}))
                if c == "inconsistent": run.state = "refuted"
                elif c == "unknown": run.state = "unknown"; run.trust = TrustLevel.UNKNOWN
            except (ValueError, TypeError) as exc:
                run.state = "unknown"; run.trust = TrustLevel.HEURISTIC
                run.result = {"consistency": "unknown", "domains": context.domains, "missing_assumptions": missing}
                run.error = str(exc)
        else:
            run.result = {"consistency": context.consistency, "domains": context.domains, "missing_assumptions": missing}
            if missing:
                run.state = "unknown"; run.trust = TrustLevel.HEURISTIC
        return run

    def _solve(self, obligation: Obligation, expr: Any, context: MathContext | None,
               artifacts: dict[str, Any]) -> ObligationExecution:
        variables = obligation.parameters.get("variables") or []
        run = ObligationExecution(obligation_id=obligation.obligation_id, kind=obligation.kind, action=obligation.action,
                                  state="succeeded", engine="sympy", trust=TrustLevel.SYMBOLIC,
                                  depends_on=obligation.depends_on, assumptions_used=_assumption_text(self.kernel, context))
        if len(variables) != 1:
            run.state = "unknown"
            run.error = "Prototype executor currently requires exactly one solve variable."
            run.result = {"variables": variables}
            return run
        variable = variables[0]
        domain = (context.domains.get(variable, "complex") if context else "complex").lower()
        try:
            sol = self.kernel.sympy.solve(expr, variable, domain)
            if not isinstance(sol, sp.FiniteSet) and domain != "complex":
                retry = self.kernel.sympy.solve(expr, variable, "complex")
                if isinstance(retry, sp.FiniteSet):
                    sol = retry
                    run.side_conditions.append(
                        f"Candidates were computed over the complex field; existence in the "
                        f"declared domain '{domain}' depends on the parameters.")
        except Exception as exc:  # SymPy may raise several solver-specific exception classes.
            run.state = "failed"; run.trust = TrustLevel.UNKNOWN; run.error = str(exc)
            return run

        # Apply contextual assumptions to explicit finite candidates.  This is an
        # important boundary: solving the equation in Reals is not enough when the
        # context also says, for example, x > 0.
        excluded=[]; candidate_conditions=[]
        if isinstance(sol, sp.FiniteSet) and context and context.assumptions:
            x=sp.Symbol(variable); accepted=[]
            for candidate in sorted(sol,key=sp.default_sort_key):
                rejected=False; unresolved=[]
                for assumption in context.assumptions:
                    a=self.kernel.sympy.to_sympy(assumption.expression)
                    check=sp.simplify(a.subs(x,candidate))
                    if check in (sp.S.false, False):
                        rejected=True
                        excluded.append({"candidate":str(candidate),"violated_assumption":render_expr(assumption.expression)})
                        break
                    if check not in (sp.S.true, True):
                        unresolved.append(str(check))
                if not rejected:
                    accepted.append(candidate)
                    if unresolved: candidate_conditions.append({"candidate":str(candidate),"conditions":unresolved})
            sol=sp.FiniteSet(*accepted)

        structured = serialize_solution_set(sol, domain)
        artifacts[obligation.obligation_id] = {"solution": sol, "variable": variable, "domain": domain,
                                                "candidate_conditions":candidate_conditions}
        run.result = {"variable": variable, "domain": domain, "solution_set": structured.model_dump(mode="json"),
                      "display": str(sol), "excluded_candidates":excluded, "candidate_conditions":candidate_conditions}
        run.side_conditions += [c for item in candidate_conditions for c in item["conditions"]]
        return run

    def _verify_candidates(self, obligation: Obligation, expr: Any, dep_runs: list[ObligationExecution],
                           artifacts: dict[str, Any]) -> ObligationExecution:
        run = ObligationExecution(obligation_id=obligation.obligation_id, kind=obligation.kind, action=obligation.action,
                                  state="unknown", engine="sympy", trust=TrustLevel.UNKNOWN, depends_on=obligation.depends_on)
        solve_id = obligation.depends_on[0]
        artifact = artifacts.get(solve_id)
        if not artifact:
            run.state = "skipped"; run.error = "No symbolic solution artifact was available."
            return run
        sol = artifact["solution"]; variable = artifact["variable"]
        if not isinstance(sol, sp.FiniteSet):
            run.result = {"checked": False, "reason": f"Candidate substitution currently supports finite sets, got {type(sol).__name__}."}
            return run
        relation = self.kernel.sympy.to_sympy(expr)
        if not isinstance(relation, sp.Equality):
            run.state = "skipped"; run.error = "Candidate verification requires an equality."
            return run
        x = sp.Symbol(variable)
        checks = []
        invalid = []
        for candidate in sorted(sol, key=sp.default_sort_key):
            residual = sp.simplify((relation.lhs - relation.rhs).subs(x, candidate))
            valid = residual == 0
            entry = {"candidate": str(candidate), "valid": bool(valid), "residual": str(residual)}
            if not valid:
                # Symbolic substitution was inconclusive; if the residual is
                # numerically ~0, spawn a dynamic interval-enclosure obligation
                # so the claim gets rigorous interval evidence mid-DAG.
                try:
                    num = complex(residual.evalf(30))
                    if abs(num) < 1e-20:
                        entry["numeric_residual"] = str(num)
                        entry["verification"] = "numeric_only"
                        run.followups.append(Obligation(
                            obligation_id=f"{obligation.obligation_id}__iv_{len(run.followups)}",
                            kind="numeric", action="interval_enclose",
                            statement=f"Interval-enclose candidate {candidate}",
                            depends_on=[obligation.obligation_id],
                            required_capabilities=["certified_enclosure"],
                            preferred_engines=["mpmath_interval"],
                            capability_ref="certified_enclosure",
                            resolved_handler="router:mpmath_interval",
                            parameters={"variable": variable, "candidate": str(candidate)}))
                except (TypeError, ValueError):
                    pass
            checks.append(entry)
            if not valid: invalid.append(entry)
        run.result = {"checked": True, "candidate_checks": checks, "invalid_candidates": invalid,
                      "all_candidates_valid": not invalid}
        run.trust = TrustLevel.SYMBOLIC
        run.state = "refuted" if invalid else "verified"
        run.evidence.append(EngineEvidence(engine="sympy", capability="symbolic_equivalence",
            status=VerificationStatus.DISPROVED if invalid else VerificationStatus.PROVED,
            trust=TrustLevel.SYMBOLIC, detail={"substitution_checks": checks}))
        artifacts[obligation.obligation_id] = {"checks": checks, "all_valid": not invalid}
        return run

    def _check_completeness(self, obligation: Obligation, expr: Any, context: MathContext | None,
                            artifacts: dict[str, Any]) -> ObligationExecution:
        run = ObligationExecution(obligation_id=obligation.obligation_id, kind=obligation.kind, action=obligation.action,
                                  state="unknown", engine="z3", trust=TrustLevel.UNKNOWN, depends_on=obligation.depends_on,
                                  assumptions_used=_assumption_text(self.kernel, context))
        solve_art = artifacts.get(obligation.depends_on[0])
        if not solve_art:
            run.state = "skipped"; run.error = "No symbolic solution artifact was available."
            return run
        sol = solve_art["solution"]
        if not isinstance(sol, sp.FiniteSet):
            run.result = {"complete": None, "reason": "SMT completeness check currently supports finite candidate sets."}
            return run
        if not self.kernel.z3.available:
            run.result = {"complete": None, "reason": "z3-solver is unavailable."}
            run.evidence.append(EngineEvidence(engine="z3", capability="smt", status="unavailable",
                                                trust=TrustLevel.UNKNOWN, error="z3-solver is not installed"))
            return run
        try:
            status, model = self.kernel.z3.find_solution_outside_candidates(
                expr, solve_art["variable"], list(sol),
                [a.expression for a in context.assumptions] if context else [],
                context.domains if context else {},
                self.kernel.sympy,
            )
        except (ValueError, TypeError) as exc:
            run.result = {"complete": None, "reason": str(exc)}
            run.evidence.append(EngineEvidence(engine="z3", capability="smt", status=VerificationStatus.UNKNOWN,
                                                trust=TrustLevel.UNKNOWN, error=str(exc), detail={"unsupported_fragment": True}))
            return run
        if status == "proved":
            run.state = "verified"; run.trust = TrustLevel.EXACT
            run.result = {"complete": True, "counterexample": None}
            run.evidence.append(EngineEvidence(engine="z3", capability="smt", status=VerificationStatus.PROVED,
                                                trust=TrustLevel.EXACT, detail={"no_solution_outside_candidates": True}))
        elif status == "disproved":
            run.state = "refuted"; run.trust = TrustLevel.EXACT
            run.result = {"complete": False, "counterexample": model}
            run.evidence.append(EngineEvidence(engine="z3", capability="smt", status=VerificationStatus.DISPROVED,
                                                trust=TrustLevel.EXACT, detail={"counterexample": model or {}}))
        else:
            run.result = {"complete": None}
        return run

    def _formalize_solution(self, obligation: Obligation, expr: Any, context: MathContext | None,
                            artifacts: dict[str, Any]) -> ObligationExecution:
        run = ObligationExecution(obligation_id=obligation.obligation_id, kind=obligation.kind, action=obligation.action,
                                  state="unknown", engine="lean", trust=TrustLevel.UNKNOWN, depends_on=obligation.depends_on,
                                  assumptions_used=_assumption_text(self.kernel, context))
        if any(self.kernel._expr_trust(ir) == TrustLevel.NUMERIC for ir in
               [expr, *([a.expression for a in context.assumptions] if context else [])]):
            run.result = {"formalized": False, "reason": "Formal proofs are disabled for approximate inputs or assumptions"}
            return run
        solve_art = artifacts.get(obligation.depends_on[0])
        if not solve_art or not isinstance(solve_art["solution"], sp.FiniteSet):
            run.result = {"formalized": False, "reason": "Formal candidate checking currently supports finite solution sets."}
            return run
        relation = self.kernel.sympy.to_sympy(expr)
        if not isinstance(relation, sp.Equality):
            run.state = "skipped"; run.error = "Formal solution soundness requires an equality."
            return run
        x = sp.Symbol(solve_art["variable"])
        certs=[]; attempts=[]; all_proved=True; any_supported=False
        for candidate in sorted(solve_art["solution"], key=sp.default_sort_key):
            try:
                residual = sp.simplify((relation.lhs - relation.rhs).subs(x, candidate))
                if residual.is_zero is False:
                    attempts.append({"candidate": str(candidate), "status": "refuted",
                                     "checked": False, "residual": str(residual)})
                    all_proved=False
                    continue
                left_ir=self.kernel.sympy.from_sympy(sp.simplify(relation.lhs.subs(x,candidate)))
                right_ir=self.kernel.sympy.from_sympy(sp.simplify(relation.rhs.subs(x,candidate)))
                status, script, tactic, error = self.kernel.lean.prove_equivalence(left_ir, right_ir, [])
                any_supported=True
                entry = {"candidate": str(candidate), "status": status, "tactic": tactic,
                         "checked": status == "proved", "error": error}
                if status == "proved":
                    certs.append({**entry, "certificate": script})
                else:
                    attempts.append({**entry, "candidate_script": script})
                    all_proved=False
            except (ValueError, TypeError) as exc:
                attempts.append({"candidate":str(candidate),"status":"unsupported","checked":False,"error":str(exc)})
                all_proved=False
        run.result={"formalized": bool(certs), "candidate_certificates": certs,
                    "candidate_attempts": attempts, "all_formally_proved": all_proved and any_supported}
        if all_proved and any_supported:
            run.state="verified"; run.trust=TrustLevel.FORMAL
            for certificate in certs:
                run.evidence.append(EngineEvidence(engine="lean", capability="formal_certificate", status=VerificationStatus.PROVED,
                    trust=TrustLevel.FORMAL, detail={"candidate": certificate["candidate"],
                                                   "certificate": certificate["certificate"]}))
        elif any(c.get("status") == "unavailable" for c in attempts):
            run.evidence.append(EngineEvidence(engine="lean", capability="formal_certificate", status="unavailable",
                                                trust=TrustLevel.UNKNOWN, error="Lean/Mathlib is unavailable; use explicit setup", detail={"candidates_generated":True}))
        return run

    def _interval_enclose(self, obligation: Obligation, expr: Any) -> ObligationExecution:
        """Rigorous interval enclosure of a numeric-only solution candidate.
        If 0 is outside the enclosure of lhs-rhs over a small interval around
        the candidate, the candidate is rigorously refuted there; containment
        is consistent-with-root evidence (interval_certified)."""
        run = ObligationExecution(obligation_id=obligation.obligation_id, kind=obligation.kind,
                                  action=obligation.action, state="unknown", engine="mpmath_interval",
                                  trust=TrustLevel.UNKNOWN, depends_on=obligation.depends_on)
        variable = obligation.parameters.get("variable")
        candidate = obligation.parameters.get("candidate")
        relation = self.kernel.sympy.to_sympy(expr)
        if not isinstance(relation, sp.Equality) or not variable or candidate is None:
            run.state = "skipped"; run.error = "interval_enclose requires an equality, a variable, and a candidate."
            return run
        try:
            import mpmath as mp
            candidate_ir = parse_math(str(candidate))
            candidate_expr = self.kernel.sympy.to_sympy(candidate_ir, {})
            if candidate_expr.free_symbols:
                raise ValueError("interval candidate must be a closed numeric expression")
            c = mp.mpf(str(sp.N(candidate_expr, 50)))
            width = mp.mpf(10) ** -30 * max(1, abs(c))
            iv = mp.iv.mpf([c - width, c + width])
            f = sp.lambdify(sp.Symbol(variable), relation.lhs - relation.rhs, modules="mpmath")
            enclosure = f(iv)
            contains = mp.mpf(0) in enclosure
            run.result = {"candidate": candidate, "enclosure": mp.nstr(enclosure, 20),
                          "contains_zero": bool(contains)}
            if contains:
                run.state = "verified"; run.trust = TrustLevel.INTERVAL_CERTIFIED
                run.evidence.append(EngineEvidence(engine="mpmath_interval", capability="interval_enclosure",
                    status=VerificationStatus.PROVED, trust=TrustLevel.INTERVAL_CERTIFIED,
                    detail={"contains_zero": True}))
            else:
                run.state = "refuted"; run.trust = TrustLevel.INTERVAL_CERTIFIED
                run.evidence.append(EngineEvidence(engine="mpmath_interval", capability="interval_enclosure",
                    status=VerificationStatus.DISPROVED, trust=TrustLevel.INTERVAL_CERTIFIED,
                    detail={"contains_zero": False}))
        except Exception as exc:
            run.error = str(exc)
        return run

    def _normalize(self, obligation: Obligation, expr: Any) -> ObligationExecution:
        run=ObligationExecution(obligation_id=obligation.obligation_id,kind=obligation.kind,action=obligation.action,
                                state="succeeded",engine="sympy",trust=TrustLevel.SYMBOLIC,depends_on=obligation.depends_on)
        try:
            out=self.kernel.sympy.simplify(expr,"simplify")
            try: display=render_expr(self.kernel.sympy.from_sympy(out))
            except ValueError: display=str(out)
            run.result={"result":display}
        except Exception as exc:
            run.state="failed"; run.trust=TrustLevel.UNKNOWN; run.error=str(exc)
        return run

    def _check_relation(self, obligation: Obligation, expr: Any, context: MathContext | None) -> ObligationExecution:
        run=ObligationExecution(obligation_id=obligation.obligation_id,kind=obligation.kind,action=obligation.action,
                                state="unknown",engine="z3",trust=TrustLevel.UNKNOWN,depends_on=obligation.depends_on,
                                assumptions_used=_assumption_text(self.kernel,context))
        if not self.kernel.z3.available:
            run.result={"satisfiable":None,"reason":"z3-solver is unavailable"}
            return run
        try:
            status, model=self.kernel.z3.check_relation(expr,[a.expression for a in context.assumptions] if context else [],context.domains if context else {})
        except (ValueError,TypeError) as exc:
            run.result={"satisfiable":None,"reason":str(exc)}; return run
        if status=="sat":
            run.state="verified";run.trust=TrustLevel.EXACT;run.result={"satisfiable":True,"model":model}
        elif status=="unsat":
            run.state="refuted";run.trust=TrustLevel.EXACT;run.result={"satisfiable":False,"model":None}
        else: run.result={"satisfiable":None,"model":None}
        return run

    def _dispatch(self, o: Obligation, expr: Any, context: MathContext | None,
                  dep_runs: list[ObligationExecution], artifacts: dict[str, Any],
                  formal: bool, missing: list[str]) -> ObligationExecution:
        """Run one obligation. Safe to call concurrently: engines are reentrant
        (fresh Z3 solvers per call, Lean via subprocess) and artifact writes use
        disjoint per-obligation keys. Derivation recording stays on the caller's
        thread to keep the DAG deterministic."""
        if (
            o.resolved_handler
            and o.resolved_handler.startswith("router:")
            and not (o.action == "formalize_solution_soundness" and not formal)
        ):
            engine_name = o.resolved_handler.removeprefix("router:")
            try:
                engine = self.kernel.router.engine(engine_name)
            except KeyError:
                return ObligationExecution(
                    obligation_id=o.obligation_id, kind=o.kind,
                    action=o.action, state="failed",
                    engine="capability_registry", trust=TrustLevel.UNKNOWN,
                    depends_on=o.depends_on,
                    error=f"Resolved engine is not registered: {engine_name}")
            if not engine.available:
                return ObligationExecution(
                    obligation_id=o.obligation_id, kind=o.kind,
                    action=o.action, state="unknown",
                    engine=engine_name, trust=TrustLevel.UNKNOWN,
                    depends_on=o.depends_on,
                    error=f"Resolved engine is unavailable: {engine_name}")
        if o.action == "validate_context":
            return self._validate_context(o, context, missing)
        if o.action == "solve_relation":
            return self._solve(o, expr, context, artifacts)
        if o.action == "verify_solution_candidates":
            return self._verify_candidates(o, expr, dep_runs, artifacts)
        if o.action == "check_solution_completeness":
            return self._check_completeness(o, expr, context, artifacts)
        if o.action == "formalize_solution_soundness":
            if formal:
                return self._formalize_solution(o, expr, context, artifacts)
            return ObligationExecution(obligation_id=o.obligation_id, kind=o.kind, action=o.action,
                                       state="skipped", engine="lean", trust=TrustLevel.UNKNOWN,
                                       depends_on=o.depends_on,
                                       error="Formal execution was disabled by the caller.")
        if o.action == "normalize_expression":
            return self._normalize(o, expr)
        if o.action == "check_relation_satisfiability":
            return self._check_relation(o, expr, context)
        if o.action == "interval_enclose":
            return self._interval_enclose(o, expr)
        return ObligationExecution(obligation_id=o.obligation_id, kind=o.kind, action=o.action,
                                   state="failed", engine="obligation_executor", trust=TrustLevel.UNKNOWN,
                                   depends_on=o.depends_on,
                                   error=f"Unknown obligation action: {o.action}")

    def execute(self, plan_id: str, plan: ProblemPlan, expr_id: str, context_id: str | None,
                *, formal: bool = True, max_steps: int = 32) -> PlanExecution:
        if max_steps < 1 or max_steps > 128:
            raise ValueError("max_steps must be between 1 and 128")
        expr=self.kernel.expressions[expr_id]
        context=self.kernel.contexts.get(context_id) if context_id else None
        by_id={o.obligation_id:o for o in plan.obligations}
        if len(by_id)!=len(plan.obligations): raise ValueError("Duplicate obligation id in plan")
        for o in plan.obligations:
            missing=[d for d in o.depends_on if d not in by_id]
            if missing: raise ValueError(f"Unknown obligation dependency: {missing[0]}")

        pending=set(by_id); completed:dict[str,ObligationExecution]={}; artifacts:dict[str,Any]={}
        sequence: list[ObligationExecution]=[]
        order={o.obligation_id: i for i, o in enumerate(plan.obligations)}
        max_dynamic = max_steps  # bound mid-DAG obligation creation
        workers = resolve_workers(None, cap=self.kernel.settings.max_workers) \
            if self.kernel.settings.enable_parallel else 1
        while pending and len(sequence)<max_steps:
            ready=[oid for oid in pending if all(d in completed for d in by_id[oid].depends_on)]
            if not ready:
                raise RuntimeError("Obligation DAG contains a cycle")
            # Stable execution order follows the planner order; dynamically
            # added obligations run after all static ones in discovery order.
            ready.sort(key=lambda oid: order[oid])
            wave=[oid for oid in ready if len(sequence)+ready.index(oid)<max_steps]

            def _run(oid: str):
                o=by_id[oid]; deps=[completed[d] for d in o.depends_on]
                if self._blocked(deps):
                    run=ObligationExecution(obligation_id=o.obligation_id,kind=o.kind,action=o.action,state="skipped",
                                            engine="obligation_executor",trust=TrustLevel.UNKNOWN,depends_on=o.depends_on,
                                            error="A dependency was failed, refuted, or skipped.")
                else:
                    run=self._dispatch(o,expr,context,deps,artifacts,formal,plan.missing_assumptions)
                return o,deps,run

            # Independent obligations in the same wave run concurrently; Z3 and
            # Lean release the GIL (C core / subprocess), so this is a real
            # speedup for verification-heavy plans. Recording stays sequential.
            for o,deps,run in thread_map(_run, wave, workers=workers):
                self._record(o,run,expr_id,deps)
                completed[o.obligation_id]=run;sequence.append(run);pending.remove(o.obligation_id)
                # Dynamic obligation creation: follow-ups discovered during
                # execution join the DAG and are scheduled in later waves.
                for f in run.followups:
                    if f.obligation_id in by_id or len(order) >= len(plan.obligations) + max_dynamic:
                        continue
                    if any(d not in by_id for d in f.depends_on):
                        continue
                    by_id[f.obligation_id]=f; order[f.obligation_id]=len(order); pending.add(f.obligation_id)

        if pending:
            for oid in list(pending):
                o=by_id[oid]
                run=ObligationExecution(obligation_id=o.obligation_id,kind=o.kind,action=o.action,state="skipped",
                                        engine="obligation_executor",trust=TrustLevel.UNKNOWN,depends_on=o.depends_on,
                                        error=f"Execution stopped at max_steps={max_steps}.")
                completed[oid]=run;sequence.append(run)

        state,status,trust,conflicts,summary=reconcile_execution(plan,sequence)
        assumptions=_assumption_text(self.kernel,context)
        side=[]
        for item in [*plan.missing_assumptions, *[c for n in sequence for c in n.side_conditions]]:
            if item not in side: side.append(item)
        graph_nodes=[];graph_edges=[]
        for n in sequence:
            graph_nodes.append({"id":n.obligation_id,"type":"obligation","action":n.action,"state":n.state,
                                "engine":n.engine,"trust":n.trust.value,"derivation_step_id":n.derivation_step_id})
            for d in n.depends_on: graph_edges.append({"from":d,"to":n.obligation_id,"type":"depends_on"})
            if n.derivation_step_id:
                graph_nodes.append({"id":n.derivation_step_id,"type":"derivation_step","operation":f"obligation:{n.action}"})
                graph_edges.append({"from":n.obligation_id,"to":n.derivation_step_id,"type":"produced_derivation"})
        graph={"nodes":graph_nodes,"edges":graph_edges}
        claim_evidence = {
            n.obligation_id: n.evidence_bundle.model_copy(deep=True)
            for n in sequence
            if not n.evidence_bundle.is_empty()
        }
        # Keep the complete obligation evidence as an audit trail, but build the
        # top-level result claim only from the evidence path that actually
        # establishes the conclusion. Optional/alternative checks must never
        # downgrade an already-established result merely because they are
        # unavailable.
        evidence_bundle = merge_evidence_bundles(
            tuple(claim_evidence.values()))
        by_action = {n.action: n for n in sequence}
        required_runs: list[ObligationExecution] = []
        if plan.classification in {"algebraic_relation", "transcendental_equation"}:
            solve = by_action.get("solve_relation")
            verify = by_action.get("verify_solution_candidates")
            complete = by_action.get("check_solution_completeness")
            if verify is not None and verify.state == "verified":
                required_runs.append(verify)
                if complete is not None and complete.state == "verified":
                    required_runs.append(complete)
                elif solve is not None:
                    required_runs.append(solve)
            elif solve is not None:
                required_runs.append(solve)
        elif plan.classification.endswith("relation"):
            sat = by_action.get("check_relation_satisfiability")
            if sat is not None:
                required_runs.append(sat)
        else:
            norm = by_action.get("normalize_expression")
            if norm is not None:
                required_runs.append(norm)

        result_parts = [
            tag_evidence_bundle(
                run.evidence_bundle, support_path="result")
            for run in required_runs if not run.evidence_bundle.is_empty()
        ]
        # If a completed/refuted result has no native bundle, retain the
        # reconciliation decision as an explicit required support record rather
        # than fabricating evidence for skipped checks.
        if not result_parts and trust != TrustLevel.UNKNOWN:
            result_parts = [EvidenceBundle(computation=[ComputationEvidence(
                engine="obligation_executor",
                method="reconcile_execution",
                arithmetic=trust.value,
                trust=trust.value,
                metadata={
                    "classification": plan.classification,
                    "final_status": status,
                },
            )])]
        result_bundle = (
            merge_evidence_bundles(tuple(result_parts))
            if result_parts else EvidenceBundle()
        )
        if not result_bundle.is_empty():
            result_bundle.justified_trust = trust.value
            claim_evidence["result"] = result_bundle
        return PlanExecution(execution_id=_eid(),plan_id=plan_id,expr_id=expr_id,context_id=context_id,state=state,
                             final_status=status,final_trust=trust,obligations=sequence,
                             propagated_assumptions=assumptions,propagated_side_conditions=side,
                             conflicts=conflicts,graph=graph,summary=summary,
                             evidence_bundle=evidence_bundle,
                             claim_evidence=claim_evidence)
