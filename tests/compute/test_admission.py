import unittest
from mathkernel_compute import ComputeRequest, CuboidParameters, ConvolutionParameters
from mathkernel_compute.models import InputBundle, ExecutionSpec, RuntimeProfile, AttemptRecord, RemoteResultEnvelope
from mathkernel_compute.protocol import digest
from mathkernel_compute.verification import verify_candidate, admit


def attempt(request):
    b = InputBundle(request=request)
    spec = ExecutionSpec(bundle=b, bundle_digest=digest(b), policy_digest='0'*64,
        runtime=RuntimeProfile(digest='1'*64,python='test',platform='test',kernel_version='test',dependencies=()))
    return AttemptRecord(attempt_id='attempt_test',workspace_id='test',execution_digest=digest(spec),bundle_digest=digest(b),spec=spec)


def candidate(a, output, **extra):
    return RemoteResultEnvelope(attempt_id=a.attempt_id,workspace_id=a.workspace_id,
        execution_digest=a.execution_digest,bundle_digest=a.bundle_digest,
        operation=a.spec.bundle.request.operation, output_schema='mk.cuboid-pairs/1' if a.spec.bundle.request.operation=='cuboid_sweep' else 'mk.real-convolution/1',
        output=output, **extra)


class AdmissionTests(unittest.TestCase):
    def test_forged_formal_metadata_cannot_affect_independent_exact_evidence(self):
        a = attempt(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='5')))
        c = candidate(a,(('3','4'),),worker_claims={'trust':'formal','verified':True,'image_hash':'forged','role':'required'})
        report = verify_candidate(a,c,'v')
        self.assertEqual(report.outcome,'PASSED')
        result = admit(a,c,report)
        self.assertEqual(result.trust.value,'exact')
        self.assertEqual(result.reconciled_trust().value,'exact')
        self.assertNotIn('forged',str(result.model_dump()))

    def test_wrong_or_incomplete_witnesses_are_rejected(self):
        a = attempt(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='5')))
        for output in [(), (('2','4'),), (('3','4'),('3','4')), (('6','8'),)]:
            c=candidate(a,output)
            self.assertEqual(verify_candidate(a,c,'v').outcome,'FAILED')
        witness=attempt(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='10'),required_claim='witnesses'))
        c=candidate(witness,(('3','4'),))
        report=verify_candidate(witness,c,'v')
        self.assertEqual(report.outcome,'PASSED')
        self.assertIn('no search completeness', report.detail)
        full=attempt(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='10')))
        self.assertEqual(verify_candidate(full,candidate(full,(('3','4'),)),'v').outcome,'FAILED')

    def test_wrong_problem_or_attempt_never_reaches_admission(self):
        a=attempt(ComputeRequest(operation='cuboid_sweep',parameters=CuboidParameters(bound='5')))
        c=candidate(a,(('3','4'),));report=verify_candidate(a,c,'v')
        for key,value in [('attempt_id','other'),('workspace_id','other'),('bundle_digest','f'*64),('execution_digest','e'*64)]:
            with self.assertRaises(ValueError):
                admit(a,c.model_copy(update={key:value}),report)
        with self.assertRaises(ValueError):
            admit(a,candidate(a,(('2','4'),)),report)

    def test_numeric_scope_and_ancestry_never_promote_to_exact(self):
        a=attempt(ComputeRequest(operation='signal_convolve',parameters=ConvolutionParameters(left=('1.5','2.5'),right=('2','-1')),required_claim='numeric_convolution'))
        c=candidate(a,('3','3.5','-2.5'),worker_claims={'trust':'interval_certified'})
        report=verify_candidate(a,c,'v')
        self.assertEqual(report.outcome,'PASSED')
        result=admit(a,c,report)
        self.assertEqual(result.trust.value,'numeric')
        self.assertEqual(result.reconciled_trust().value,'numeric')
        self.assertIn('binary64 inputs',report.detail)
        self.assertEqual(verify_candidate(a,candidate(a,('3','30','-2.5')),'v').outcome,'FAILED')
