"""Requires the core runtime; skipped explicitly when its dependencies are absent."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ('sympy','z3','numpy','pydantic','lark','mpmath'))


@unittest.skipUnless(AVAILABLE, 'Full MathKernel dependencies are not installed in this environment')
class KernelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mathkernel import MathKernel
        cls.kernel=MathKernel()

    def test_live_catalog_uses_real_capability_query_without_ports_or_execution(self):
        from mathkernel_studio.source import KernelSource
        source=KernelSource(self.kernel,host_id='integration-host',workspace_id='integration-workspace',version='source-test')
        self.assertTrue(source.entries)
        self.assertTrue(all(e['composition']=='partial' and e['input_ports']==[] and e['output_ports']==[] for e in source.entries))
        self.assertFalse(source.handshake()['features']['workflow_execute'])

    def test_existing_result_evidence_and_exact_text_survive_inspection_snapshot(self):
        from mathkernel_studio.source import KernelSource, InspectionRecord
        result=self.kernel.parse('9007199254740993')
        self.assertTrue(result.ok)
        record=InspectionRecord.from_result(result,result_ref='r',source_ref='existing-expression',source_revision='1')
        source=KernelSource(self.kernel,host_id='integration-host',workspace_id='integration-workspace',records=(record,),version='source-test')
        output=source.result('r')
        self.assertEqual(output['result'],result.model_dump(mode='json'))
        self.assertEqual(output['claim_trust'],{name:b.conservative_trust() for name,b in result.claim_evidence.items()})
        before=source.result('r');result.data['later_mutation']='must not appear in snapshot'
        self.assertEqual(source.result('r'),before)

    def test_paged_receipt_is_resolved_only_from_registered_result_reference(self):
        from mathkernel.models import MathResult
        from mathkernel.output_policy import enforce_output_budget
        from mathkernel_studio.source import KernelSource, InspectionRecord
        old=self.kernel.settings.max_output_size_bytes
        try:
            source=KernelSource(self.kernel,host_id='integration-host',workspace_id='integration-workspace',version='source-test')
            self.kernel.settings.max_output_size_bytes=1000
            receipt=enforce_output_budget(self.kernel,MathResult(ok=True,data={'payload':'x'*10000}))
            record=InspectionRecord.from_result(receipt,result_ref='receipt',source_ref='stored-payload',source_revision='1')
            source.records={'receipt':record}
            self.assertEqual(source.result('receipt')['claim_trust'],{})
            first=source.result_page('receipt',0);self.assertEqual(first['resource_id'],record.resource_id)
            with self.assertRaises(KeyError):source.result_page('arbitrary-resource',0)
        finally:self.kernel.settings.max_output_size_bytes=old


if __name__=='__main__': unittest.main()
