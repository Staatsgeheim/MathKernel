# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Restricted MathIR adapter for certificate-aware MPC."""
from . import mpc
from .engineering import positive_integer

OPERATIONS={
    'StateSpaceSystem':{'mpc':{
        **{key:'MathIR[][]' for key in ('Q','R','terminal')},'initial':'MathIR[]',
        **{key:'MathIR?[]' for key in ('state_lower','state_upper','input_lower','input_upper','terminal_lower','terminal_upper')},
        'terminal_gain':'MathIR[][]?','horizon':'positive integer',
        'certificate_id':'OptimizationCertificate object id?','tolerance':'positive finite tolerance',
        'max_iterations':'positive integer','reconstruction_denominator':'positive integer'}},
    'MPCPlan':{'verify':{'tolerance':'positive finite tolerance'},
               'first_control':{'tolerance':'positive finite tolerance'}},
}


def apply(kernel,value,object_type,operation,p,reference,object_id):
    import math
    from .engineering_adapter import _parse_tree
    s=kernel.settings
    if object_type=='StateSpaceSystem':system=value
    else:
        record=kernel._get_math_object_record(value.system_id)
        if record is None or record['object_type']!='StateSpaceSystem':
            raise ValueError('source state-space system is unavailable')
        system=record['value']
    tolerance=p.get('tolerance',1e-9)
    if isinstance(tolerance,bool) or not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or not 0<tolerance<1:
        raise ValueError('tolerance must be finite and in (0,1)')
    if object_type=='MPCPlan':
        if operation=='first_control':return mpc.first_control(value,system,tolerance)
        report,outputs=mpc.verify_plan(value,system,tolerance=tolerance)
        return (report,outputs) if outputs else report
    horizon=positive_integer(p.get('horizon'), 'horizon', s.max_mpc_horizon)
    n,m=len(system.A),len(system.B[0]);variables=(horizon+1)*n+horizon*m
    constraints=(horizon+1)*n+2*variables # equality rows plus worst-case two-sided bounds
    if variables>s.max_optimization_variables or constraints>s.max_optimization_constraints or variables**3>s.max_engineering_work:
        raise ValueError('MPC transcription exceeds optimization/work limits')
    fields={key:p[key] for key,kind in OPERATIONS['StateSpaceSystem']['mpc'].items()
            if 'MathIR' in kind and key in p and p[key] is not None}
    for key,item in fields.items():
        if not isinstance(item,(list,tuple)) or len(item)>s.max_control_order:
            raise ValueError(f'{key} exceeds control dimension limit')
        if '[][]' in OPERATIONS['StateSpaceSystem']['mpc'][key] and any(not isinstance(row,(list,tuple)) or len(row)>s.max_control_order for row in item):
            raise ValueError(f'{key} exceeds control dimension limit')
    parsed,trust=_parse_tree(kernel,fields,p.get('context_id'))
    cert=None;cert_trust='exact'
    if 'certificate_id' in p:
        cert=reference('certificate_id','OptimizationCertificate')
        cert_trust=kernel._get_math_object_record(str(p['certificate_id']))['input_trust'].value
    return mpc.synthesize(system,object_id,horizon=horizon,mode=p.get('mode','exact'),
        certificate=cert,certificate_trust=cert_trust,input_trust=trust.value,tolerance=tolerance,
        max_iterations=positive_integer(p.get('max_iterations',s.max_iterations),'max_iterations',s.max_iterations),
        time_limit=s.solver_timeout_seconds*.9,
        reconstruction_denominator=positive_integer(p.get('reconstruction_denominator',1_000_000),'reconstruction_denominator',1_000_000_000),
        **parsed)
