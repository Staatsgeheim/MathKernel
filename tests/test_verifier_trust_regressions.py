"""Regressions for the public trust teardown and adjacent evidence paths."""
import pytest

from mathkernel import MathKernel, TrustLevel
from mathkernel.models import DerivationStep, EngineEvidence, MathResult, ObligationExecution
from mathkernel_artifacts import ComputationEvidence, EvidenceBundle


@pytest.mark.parametrize('formal', [False, True])
def test_declined_verifiers_do_not_erase_symbolic_identity(formal):
    result = MathKernel().prove_equivalence('sin(x)^2+cos(x)^2', '1', formal=formal)
    assert result.status == 'verified'
    assert result.trust == result.reconciled_trust() == TrustLevel.SYMBOLIC
    assert result.semantic_status.value == 'verified_symbolic'
    assert result.derivation[0].claim_evidence['result'].conservative_trust() == 'symbolic'
    assert all(e.role == 'diagnostic' for e in result.evidence if e.status == 'unknown')


def test_refutation_has_exact_support_and_no_unchecked_proof(monkeypatch):
    kernel = MathKernel()
    def forbidden(*args, **kwargs):
        pytest.fail('Lean must not run after an exact refutation')
    monkeypatch.setattr(kernel.lean, 'prove_equivalence', forbidden)
    result = kernel.prove_equivalence('(x+y)^2', 'x^2+y^2')
    assert result.status == 'refuted'
    assert result.trust == result.reconciled_trust() == TrustLevel.EXACT
    assert result.data['counterexample']
    assert 'lean_certificate' not in result.data and 'lean_candidate' not in result.data
    assert not result.evidence_bundle.proof


@pytest.mark.parametrize('status', ['unavailable', 'error'])
def test_unchecked_scripts_are_only_explicit_candidates(monkeypatch, status):
    kernel = MathKernel()
    monkeypatch.setattr(kernel.lean, 'prove_equivalence', lambda *args: (status, 'unchecked script', 'ring', 'not checked'))
    result = kernel.prove_equivalence('x+x', '2*x')
    assert result.trust == TrustLevel.EXACT
    assert result.reconciled_trust() == TrustLevel.EXACT
    assert 'lean_certificate' not in result.data
    assert result.data['lean_candidate'] == dict(script='unchecked script', tactic='ring', checked=False, status=status)
    assert not any(p.engine == 'lean' for p in result.evidence_bundle.proof)


def test_checked_certificate_binds_to_formal_evidence(monkeypatch):
    kernel = MathKernel()
    script = 'checked fixture'
    monkeypatch.setattr(kernel.lean, 'prove_equivalence', lambda *args: ('proved', script, 'ring', None))
    result = kernel.prove_equivalence('x+x', '2*x')
    assert result.trust == result.reconciled_trust() == TrustLevel.FORMAL
    assert result.semantic_status.value == 'proved'
    assert result.data['lean_certificate'] == script
    proof = next(p for p in result.evidence_bundle.proof if p.engine == 'lean')
    assert proof.certificate == script and proof.verified
    assert proof.support_path == 'verifier:lean'


@pytest.mark.parametrize('left,right', [('0.7+0.3', '1'), ('0.1+0.2', '0.3')])
def test_decimal_claim_and_derivation_never_contain_exact_proofs(left, right, monkeypatch):
    kernel = MathKernel()
    def forbidden(*args, **kwargs):
        pytest.fail('Exact/formal backends must not rationalize approximate inputs')
    monkeypatch.setattr(kernel.z3, 'counterexample_equivalence', forbidden)
    monkeypatch.setattr(kernel.lean, 'prove_equivalence', forbidden)
    result = kernel.prove_equivalence(left, right)
    assert result.trust in (TrustLevel.NUMERIC, TrustLevel.UNKNOWN)
    assert result.reconciled_trust() == result.trust
    for value in [result, *result.derivation]:
        assert value.claim_evidence['result'].conservative_trust() == result.trust.value
        assert all(p.trust == 'numeric' for p in value.evidence_bundle.proof)
    assert 'lean_certificate' not in result.data
    restored = MathResult.model_validate_json(result.model_dump_json())
    assert restored.reconciled_trust() == restored.trust


def test_decimal_assumptions_disable_exact_proof_routes():
    kernel = MathKernel()
    context = kernel.create_context({'x': 'real'}, ['x > 0.1'])
    result = kernel.prove_equivalence('x+x', '2*x', context.context_id)
    assert result.trust == result.reconciled_trust() == TrustLevel.NUMERIC
    assert not any(e.engine in {'z3', 'lean'} and e.status == 'proved' for e in result.evidence)
    counterexample = kernel.counterexample('x', '0', context.context_id)
    assert counterexample.status == 'unknown'
    goal = kernel.parse('x > 0').data['expr_id']
    assert kernel.prove(goal, context.context_id).status == 'unknown'


@pytest.mark.parametrize('source', ['0.1+0.2 = 0.3', '0.7+0.3 = 1'])
def test_general_and_batch_provers_refuse_approximate_semantics(source):
    kernel = MathKernel()
    goal = kernel.parse(source).data['expr_id']
    result = kernel.prove(goal)
    assert result.status == 'unknown'
    assert 'certificate' not in result.data and 'certificate_id' not in result.data
    batch = kernel.prove_batch([goal], workers=1)
    assert batch.trust == TrustLevel.UNKNOWN
    assert batch.data['results'][0]['status'] == 'unknown'


def test_batch_proof_preserves_context_assumptions():
    kernel = MathKernel()
    context = kernel.create_context({'x': 'real'}, ['x > 0'])
    goal = kernel.parse('x > 0').data['expr_id']
    result = kernel.prove_batch([goal], context.context_id, workers=1)
    assert result.data['results'][0]['status'] == 'valid'
    assert result.trust == TrustLevel.EXACT


@pytest.mark.parametrize('factory', [
    lambda e: MathResult(ok=True, trust=TrustLevel.EXACT, evidence=e),
    lambda e: DerivationStep(step_id='s', operation='verify', evidence=e),
    lambda e: ObligationExecution(obligation_id='o', kind='formal', action='verify', evidence=e),
])
@pytest.mark.parametrize('status', ['unknown', 'unavailable', 'error'])
def test_legacy_projection_preserves_required_unknown_dependencies(factory, status):
    records = [EngineEvidence(engine='checker', capability='verify', status='proved', trust='exact'),
               EngineEvidence(engine='input', capability='ancestry', status=status, trust='exact')]
    value = factory(records)
    assert value.evidence_bundle.conservative_trust() == 'unknown'
    records[1].role = 'diagnostic'
    assert factory(records).evidence_bundle.conservative_trust() == 'exact'


def test_unrelated_claims_keep_their_own_trust():
    claims = {name: EvidenceBundle(computation=[ComputationEvidence(
        engine='fixture', method=name, arithmetic=trust, trust=trust)])
        for name, trust in [('construction', 'exact'), ('estimate', 'numeric')]}
    result = MathResult(ok=True, trust='numeric', claim_evidence=claims)
    assert result.reconciled_trust(['construction']) == TrustLevel.EXACT
    assert result.reconciled_trust() == TrustLevel.NUMERIC


@pytest.mark.parametrize('method', ['equivalence', 'relation'])
def test_lean_adapter_itself_refuses_approximate_claims(method, monkeypatch):
    from mathkernel.engines import LeanEngine
    from mathkernel.parser import parse_math
    engine = LeanEngine()
    def forbidden(*args, **kwargs): pytest.fail('Approximate goal reached Lean')
    monkeypatch.setattr(engine, '_check_script', forbidden)
    with pytest.raises(ValueError, match='approximate'):
        if method == 'equivalence':
            engine.prove_equivalence(parse_math('0.1+0.2'), parse_math('0.3'))
        else:
            engine.prove_relation(parse_math('0.7+0.3=1'))


@pytest.mark.parametrize('status', ['proved', 'unavailable', 'error'])
def test_solution_reasoning_separates_checked_certificates_from_attempts(monkeypatch, status):
    import sympy as sp
    from mathkernel.execution import ObligationExecutor
    from mathkernel.models import Obligation
    from mathkernel.parser import parse_math
    kernel = MathKernel()
    monkeypatch.setattr(kernel.lean, 'prove_equivalence', lambda *args: (status, 'fixture script', 'norm_num', None))
    obligation = Obligation(obligation_id='formal', kind='formal',
        action='formalize_solution_soundness', statement='candidate soundness', depends_on=['solve'])
    run = ObligationExecutor(kernel)._formalize_solution(obligation, parse_math('x=1'), None,
        {'solve': {'solution': sp.FiniteSet(1), 'variable': 'x'}})
    if status == 'proved':
        assert run.result['candidate_certificates'][0]['certificate'] == 'fixture script'
        assert run.result['candidate_certificates'][0]['checked'] is True
        assert run.result['candidate_attempts'] == []
        projected = ObligationExecution.model_validate(run.model_dump())
        assert projected.evidence_bundle.proof[0].certificate == 'fixture script'
    else:
        assert run.result['candidate_certificates'] == []
        assert run.result['candidate_attempts'][0]['checked'] is False
        assert 'certificate' not in run.result['candidate_attempts'][0]


def test_solution_reasoning_never_generates_proof_of_refuted_candidate(monkeypatch):
    import sympy as sp
    from mathkernel.execution import ObligationExecutor
    from mathkernel.models import Obligation
    from mathkernel.parser import parse_math
    kernel = MathKernel()
    def forbidden(*args, **kwargs): pytest.fail('Refuted candidate reached Lean')
    monkeypatch.setattr(kernel.lean, 'prove_equivalence', forbidden)
    obligation = Obligation(obligation_id='formal', kind='formal',
        action='formalize_solution_soundness', statement='candidate soundness', depends_on=['solve'])
    run = ObligationExecutor(kernel)._formalize_solution(obligation, parse_math('x=1'), None,
        {'solve': {'solution': sp.FiniteSet(0), 'variable': 'x'}})
    assert run.result['candidate_certificates'] == []
    assert run.result['candidate_attempts'][0]['status'] == 'refuted'
    assert 'candidate_script' not in run.result['candidate_attempts'][0]
