"""Optional presentation adapter boundary, not a workflow runtime.

Only an explicitly supplied service can enable these routes. The service owns
admissibility, immutable plans, durable request reconciliation and authorization.
The default KernelSource has no service and cannot invoke these operations.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Protocol
from .source import check_id

ACTION_FIELDS = {
    'workflow/validate': {'document', 'binding'},
    'workflow/plan': {'validation_ref', 'binding', 'scope', 'selected_nodes'},
    'workflow/submit': {'plan_ref', 'plan_digest', 'authority_ref', 'client_request_id'},
    'approval/challenge': {'plan_ref', 'plan_digest'},
    'approval/confirm': {'challenge_ref', 'plan_ref', 'plan_digest', 'decision'},
    'runs/action': {'run_ref', 'revision', 'action', 'client_request_id'},
}
ACTION_FEATURES = {'workflow/validate': 'workflow_validate', 'workflow/plan': 'workflow_execute',
    'workflow/submit': 'workflow_execute', 'approval/challenge': 'approval_interact',
    'approval/confirm': 'approval_interact', 'runs/action': 'run_observe'}

@dataclass(frozen=True)
class SessionScope:
    host_instance_id: str
    workspace_id: str
    session_id: str

class WorkflowPresentationService(Protocol):
    def capabilities(self) -> dict: ...
    def command(self, action: str, payload: dict, request_id: str, scope: SessionScope) -> dict: ...
    def read(self, kind: str, reference: str | None, offset: int, scope: SessionScope) -> dict: ...

def parse_command(data: bytes, action: str) -> dict:
    """Bounded JSON framing; semantic validation remains mandatory in the service."""
    if action not in ACTION_FIELDS or not 0 < len(data) <= 10 * 1024 * 1024 + 8192:
        raise ValueError('Unsupported command or byte budget')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result or key in {'__proto__', 'constructor', 'prototype'}:
                raise ValueError('Duplicate or unsafe key')
            result[key] = value
        return result
    def integer(text):
        if len(text) > 17 or abs(int(text)) > 2**53-1: raise ValueError('Unsafe numeric literal')
        return int(text)
    def floating(text):
        from decimal import Decimal
        number = Decimal(text)
        if not number.is_finite() or abs(number) > 2**53-1: raise ValueError('Unsafe numeric literal')
        return float(number)
    def constant(_): raise ValueError('Nonfinite JSON')
    value = json.loads(data, object_pairs_hook=pairs, parse_int=integer, parse_float=floating, parse_constant=constant)
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop(); count += 1
        if depth > 48 or count > 300000: raise ValueError('Structure exceeds budget')
        if isinstance(item, dict): pending.extend((v, depth+1) for v in item.values())
        elif isinstance(item, list): pending.extend((v, depth+1) for v in item)
    if not isinstance(value, dict) or set(value) != ACTION_FIELDS[action]: raise ValueError('Invalid command fields')
    for key in ('validation_ref', 'plan_ref', 'challenge_ref', 'run_ref', 'client_request_id', 'authority_ref'):
        if key in value and value[key] is not None: check_id(value[key])
    if 'plan_digest' in value:
        import re
        if not isinstance(value['plan_digest'], str) or not re.fullmatch('[0-9a-f]{64}', value['plan_digest']): raise ValueError('Invalid digest')
    if 'decision' in value and value['decision'] not in ('approve', 'deny'): raise ValueError('Invalid decision')
    if action == 'runs/action' and value['action'] not in ('cancel', 'retry', 'reverify', 'resume'): raise ValueError('Unsupported run action')
    return value
