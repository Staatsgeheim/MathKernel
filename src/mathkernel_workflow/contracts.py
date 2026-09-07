"""Explicit, versioned composition contracts for existing kernel entrypoints."""
from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass
VERSION = 'local-workflow/1'
ID = re.compile('^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$')

def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {'constructor', 'prototype', '__proto__'}:
        raise ValueError('Invalid identifier')
    return value

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf8')

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def binding(document):
    semantic = {'document_id': document['identity']['document_id'], **document['authoring'], 'nodes': [{**n, 'label': ''} for n in document['authoring']['nodes']], 'host_binding': document.get('host_binding')}
    return {'document_id': document['identity']['document_id'], 'draft_revision': document['authoring']['revision'], 'document_digest': digest(semantic)}

@dataclass(frozen=True)
class Operation:
    method: str
    inputs: tuple[tuple[str, str], ...]
    output: str
    output_key: str | None
    parameters: tuple[tuple[str, str, object], ...] = ()
OPERATIONS = {'parse': Operation('parse', (), 'Expression', 'expr_id', (('expression', 'string', None),)), 'simplify': Operation('simplify', (('expr_id', 'Expression'),), 'Expression', 'result_expr_id', (('mode', 'string', 'simplify'),)), 'differentiate': Operation('differentiate', (('expr_id', 'Expression'),), 'Expression', 'result_expr_id', (('variable', 'string', None), ('order', 'integer', 1))), 'integrate': Operation('integrate', (('expr_id', 'Expression'),), 'Expression', 'result_expr_id', (('variable', 'string', None),)), 'solve': Operation('solve', (('expr_id', 'Expression'),), 'Result', None, (('variable', 'string', None),)), 'analyze': Operation('analyze', (('expr_id', 'Expression'),), 'Result', None), 'matrix_create': Operation('matrix_create', (), 'Matrix', 'matrix_id', (('rows', 'array', None),)), 'matrix_det': Operation('matrix_det', (('matrix_id', 'Matrix'),), 'Expression', 'result_expr_id'), 'matrix_inverse': Operation('matrix_inverse', (('matrix_id', 'Matrix'),), 'Matrix', 'matrix_id'), 'matrix_transpose': Operation('matrix_transpose', (('matrix_id', 'Matrix'),), 'Matrix', 'matrix_id'), 'matrix_multiply': Operation('matrix_multiply', (('left_id', 'Matrix'), ('right_id', 'Matrix')), 'Matrix', 'matrix_id'), 'matrix_rank': Operation('matrix_rank', (('matrix_id', 'Matrix'),), 'Result', None), 'matrix_rref': Operation('matrix_rref', (('matrix_id', 'Matrix'),), 'Matrix', 'matrix_id'), 'matrix_eigenvalues': Operation('matrix_eigenvalues', (('matrix_id', 'Matrix'),), 'Result', None), 'matrix_solve': Operation('matrix_solve', (('matrix_id', 'Matrix'), ('rhs_id', 'Matrix')), 'Matrix', 'matrix_id')}

def descriptors():
    out = []
    for name, op in OPERATIONS.items():
        ref = 'workflow.' + name

        def port(key, kind, direction):
            return dict(id=key, direction=direction, label=key, type_ref=kind, cardinality='one', required=True, shape=None)
        properties = {key: {'type': kind, **({'default': default} if default is not None else {})} for key, kind, default in op.parameters}
        schema = {'type': 'object', 'properties': properties, 'required': [key for key, _, default in op.parameters if default is None], 'additionalProperties': False}
        contract = {'method': op.method, 'inputs': op.inputs, 'output': op.output, 'key': op.output_key, 'parameters': op.parameters, 'version': VERSION}
        out.append(dict(descriptor_id=ref, operation_ref=ref, operation_version=VERSION, schema_digest=digest(contract), title=name.replace('_', ' ').title(), domain='Executable local workflows', description=f'Existing MathKernel.{op.method}; isolated local CPU process. No remote export.', availability='available', composition='complete', input_ports=[port(k, t, 'input') for k, t in op.inputs], output_ports=[port('value', op.output, 'output')], input_types=[t for _, t in op.inputs], output_types=[op.output], engines=['mathkernel'], trust_levels=[], verification_methods=[], parameter_schema=schema, coverage='Explicit local adapter. Trust and semantic status come only from the returned MathResult.'))
    return out
CATALOG = {d['operation_ref']: d for d in descriptors()}

def parameters(node, op):
    drafts = node['parameter_drafts']
    if set(drafts) - {k for k, _, _ in op.parameters}:
        raise ValueError('Unknown operation parameter')
    out = {}
    for key, kind, default in op.parameters:
        if key not in drafts:
            if default is None:
                raise ValueError(f'Missing parameter {key}')
            out[key] = default
            continue
        field = drafts[key]
        if set(field) != {'encoding', 'text'} or field['encoding'] not in {'text', 'json_text'} or (not isinstance(field['text'], str)):
            raise ValueError('Invalid parameter draft')
        value = json.loads(field['text']) if field['encoding'] == 'json_text' else field['text']
        if kind == 'string' and (not isinstance(value, str) or not value or len(value) > 16384):
            raise ValueError(f'{key} requires nonempty text within 16 KiB')
        if kind == 'integer':
            if isinstance(value, str) and re.fullmatch('[0-9]{1,3}', value):
                value = int(value)
            if type(value) is not int or not 1 <= value <= 64:
                raise ValueError(f'{key} requires integer 1–64')
        if kind == 'array':
            validate_rows(value)
        if key == 'variable' and (not re.fullmatch('[A-Za-z][A-Za-z0-9_]{0,63}', value)):
            raise ValueError('Variable must be an identifier')
        if key == 'mode' and value not in {'simplify', 'expand', 'factor', 'cancel', 'trig', 'rational', 'normal_form'}:
            raise ValueError('Unsupported simplification mode')
        out[key] = value
    return out

def validate_rows(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 32 or (not isinstance(rows[0], list)) or (not 1 <= len(rows[0]) <= 32):
        raise ValueError('Matrix must be 1–32 by 1–32')
    if any((not isinstance(row, list) or len(row) != len(rows[0]) or any((not isinstance(v, str) or not v or len(v) > 1024 for v in row)) for row in rows)):
        raise ValueError('Matrix cells must be rectangular mathematical text')

def input_value(node):
    if set(node['parameter_drafts']) != {'value'}:
        raise ValueError('Input requires a value draft')
    field = node['parameter_drafts']['value']
    if field['encoding'] != 'json_text':
        raise ValueError('Input requires structured JSON text')
    v = json.loads(field['text'])
    kind = v['kind']
    if kind == 'matrix':
        if set(v) != {'kind', 'arithmetic', 'cells'} or v['arithmetic'] != 'exact':
            raise ValueError('Only exact matrix inputs are supported by the local adapter')
        validate_rows(v['cells'])
        return ('Matrix', v['cells'])
    if kind == 'integer':
        if set(v) != {'kind', 'value'} or not isinstance(v['value'], str) or (not re.fullmatch('-?[0-9]{1,4096}', v['value'])):
            raise ValueError('Invalid exact integer')
        return ('Expression', v['value'])
    if kind == 'rational':
        if set(v) != {'kind', 'numerator', 'denominator'} or any((not isinstance(v[k], str) or not re.fullmatch('-?[0-9]{1,4096}', v[k]) for k in ['numerator', 'denominator'])) or int(v['denominator']) == 0:
            raise ValueError('Invalid exact rational')
        return ('Expression', f"({v['numerator']})/({v['denominator']})")
    if kind == 'expression':
        if set(v) != {'kind', 'language', 'text', 'context'} or v['language'] != 'mathkernel' or v['context'] or (not isinstance(v['text'], str)) or (not 1 <= len(v['text']) <= 16384):
            raise ValueError('Only context-free MathKernel expression text is supported')
        return ('Expression', v['text'])
    raise ValueError('Unsupported input kind; numerical inputs need an explicit numerical adapter')
