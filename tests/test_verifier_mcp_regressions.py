"""Trust and proof-artifact semantics survive the actual MCP serialization."""
import asyncio

import pytest


def test_verifier_results_keep_their_evidence_contract_over_mcp(monkeypatch):
    pytest.importorskip('fastmcp')
    from fastmcp import Client
    from mathkernel import MathKernel
    from mathkernel import lean_bootstrap
    from mathkernel_mcp import server

    monkeypatch.setattr(lean_bootstrap, 'resolve_lean_toolchain', lambda: None)
    monkeypatch.setattr(server, 'kernel', MathKernel())

    async def check():
        async with Client(server.mcp) as client:
            for left, right, trust, status in (
                ('sin(x)^2+cos(x)^2', '1', 'symbolic', 'verified'),
                ('(x+y)^2', 'x^2+y^2', 'exact', 'refuted'),
                ('0.7+0.3', '1', 'numeric', 'verified'),
            ):
                result = (await client.call_tool('math_prove_equivalence',
                    {'left': left, 'right': right, 'formal': True})).data
                assert result['trust'] == trust and result['status'] == status
                assert 'lean_certificate' not in result['data']
                if status == 'refuted':
                    assert 'lean_candidate' not in result['data']
                assert all(p['trust'] == trust for p in result['evidence_bundle']['proof'])
    asyncio.run(check())
