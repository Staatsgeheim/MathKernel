# =============================================================================
# MathKernel - decimal fuzz persistence tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import os, random, tempfile
from mathkernel.kernel import MathKernel
from mathkernel.models import TrustLevel
from mathkernel.parser import parse_math
from mathkernel.store import KernelStore

def test_decimal_trust_fuzz_250_trees():
    rng=random.Random(0xDEC1A1); k=MathKernel()
    for _ in range(250):
        d=rng.choice(['0.1','0.2','0.7','0.001','1.5','12.3'])
        x=rng.choice(['x','y','(x+1)','(x*x+1)','(y-2)'])
        expr=rng.choice([f'({x})+({d})',f'({d})*({x})',f'(({x})+{d})/(({x})*({x})+1)',f'({d})-({d})+({x})'])
        p=k.parse(expr); assert p.trust==TrustLevel.NUMERIC
        assert k.simplify(p.data['expr_id']).trust==TrustLevel.NUMERIC

def test_decimal_store_roundtrip_preserves_literal_and_precision():
    fd,path=tempfile.mkstemp(suffix='.sqlite'); os.close(fd)
    try:
        store=KernelStore(path); ir=parse_math('0.700*x+1')
        store.put_expression('e',{'ir':ir.model_dump(mode='json'),'source':'0.700*x+1'})
        got=store.get_expression('e'); store.close()
        real=got['ir']['args'][0]['args'][0]
        assert real['kind']=='real' and real['value']=='0.700' and real['precision']==3
    finally:
        os.unlink(path)
