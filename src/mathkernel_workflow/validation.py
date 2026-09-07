"""Structural/type validation; mathematics is still checked by kernel execution."""
from __future__ import annotations
import copy
from .contracts import CATALOG, OPERATIONS, binding, canonical, identifier, input_value, parameters, digest

def compile_document(document, expected, max_nodes=100, subworkflows=None):
    subworkflows = subworkflows or {}
    if len(canonical(document)) > 10 * 1024 * 1024:
        raise ValueError('Document byte budget exceeded')
    if set(document) != {'schema', 'identity', 'authoring', 'presentation', 'host_binding'} or document['schema'] != 'mk.studio/1':
        raise ValueError('Unsupported document schema')
    a = document['authoring']
    if set(a) != {'revision', 'nodes', 'edges', 'desired_outputs'} or type(a['revision']) is not int or (not 0 <= a['revision'] < 2 ** 53):
        raise ValueError('Invalid authoring revision')
    if set(document['identity']) != {'document_id', 'title'} or not isinstance(document['identity']['title'], str) or (not 1 <= len(document['identity']['title']) <= 8192):
        raise ValueError('Invalid document identity/title')
    identifier(document['identity']['document_id'])
    if binding(document) != expected:
        raise ValueError('Frozen document digest or revision mismatch')
    if not isinstance(a['nodes'], list) or not 1 <= len(a['nodes']) <= max_nodes or (not isinstance(a['edges'], list)) or (len(a['edges']) > max_nodes * 4):
        raise ValueError('Local execution graph budget exceeded')
    nodes = {}
    types = {}
    compiled = {}
    for node in a['nodes']:
        if set(node) != {'id', 'kind', 'label', 'operation_ref', 'operation_version', 'schema_digest', 'parameter_drafts', 'input_bindings'}:
            raise ValueError('Unsupported node fields')
        key = identifier(node['id'])
        if key in nodes:
            raise ValueError('Duplicate node ID')
        if node['input_bindings']:
            raise ValueError('External object bindings require a separately authorized object adapter')
        nodes[key] = node
        if node['kind'] == 'object_binding' and node['operation_ref'] is None:
            kind, value = input_value(node)
            types[key] = kind
            compiled[key] = {'id': key, 'method': 'matrix_create' if kind == 'Matrix' else 'parse', 'parameters': {'rows' if kind == 'Matrix' else 'expression': value}, 'inputs': {}, 'output_key': 'matrix_id' if kind == 'Matrix' else 'expr_id'}
        elif node['kind'] in {'operation', 'subworkflow'} and node['operation_ref'] in subworkflows:
            saved = subworkflows[node['operation_ref']]
            if node['operation_version'] != saved['public']['revision'] or node['schema_digest'] != saved['public']['digest'] or node['parameter_drafts']:
                raise ValueError('Saved workflow revision/digest mismatch or unsupported parameters')
            types[key] = saved['output_type']
            compiled[key] = {'id': key, 'method': 'subworkflow', 'inputs': {}, 'nested': saved['operations'], 'root': saved['root']}
        elif node['kind'] == 'operation' and node['operation_ref'] in CATALOG:
            contract = CATALOG[node['operation_ref']]
            if node['operation_version'] != contract['operation_version'] or node['schema_digest'] != contract['schema_digest']:
                raise ValueError('Operation descriptor changed; review/rebind it')
            op = OPERATIONS[node['operation_ref'].removeprefix('workflow.')]
            types[key] = op.output
            compiled[key] = {'id': key, 'method': op.method, 'parameters': parameters(node, op), 'inputs': {}, 'output_key': op.output_key}
        else:
            raise ValueError('Unsupported operation or control construct: ' + str(node['operation_ref']))
    edges = set()
    incoming = {key: set() for key in nodes}
    outgoing = {key: [] for key in nodes}
    for edge in a['edges']:
        if set(edge) != {'id', 'source_node', 'source_port', 'target_node', 'target_port'}:
            raise ValueError('Unsupported edge fields')
        eid = identifier(edge['id'])
        if eid in edges:
            raise ValueError('Duplicate edge ID')
        edges.add(eid)
        source = edge['source_node']
        target = edge['target_node']
        port = edge['target_port']
        if source not in nodes or target not in nodes or edge['source_port'] != 'value':
            raise ValueError('Unknown node or source port')
        spec = OPERATIONS.get(nodes[target]['operation_ref'].removeprefix('workflow.')) if nodes[target]['operation_ref'] else None
        ports = dict(spec.inputs) if spec else {}
        if port not in ports or ports[port] != types[source]:
            raise ValueError('Port type mismatch or unknown target port')
        if port in compiled[target]['inputs']:
            raise ValueError('Occupied input port')
        compiled[target]['inputs'][port] = source
        incoming[target].add(source)
        outgoing[source].append(target)
    for key, node in nodes.items():
        op = OPERATIONS.get(node['operation_ref'].removeprefix('workflow.')) if node['operation_ref'] else None
        if op and set(compiled[key]['inputs']) != {k for k, _ in op.inputs}:
            raise ValueError('Required input connection missing for ' + key)
    degrees = {k: len(v) for k, v in incoming.items()}
    queue = [k for k in nodes if not degrees[k]]
    ordered = []
    for key in queue:
        ordered.append(compiled[key])
        for target in set(outgoing[key]):
            degrees[target] -= 1
            if not degrees[target]:
                queue.append(target)
    if len(ordered) != len(nodes):
        raise ValueError('Cycles are not supported')
    outputs = []
    if not isinstance(a['desired_outputs'], list):
        raise ValueError('Invalid outputs')
    for out in a['desired_outputs']:
        if set(out) != {'node_id', 'port_id'} or out['node_id'] not in nodes or out['port_id'] != 'value':
            raise ValueError('Unknown desired output')
        if out['node_id'] in outputs:
            raise ValueError('Duplicate desired output')
        outputs.append(out['node_id'])
    expanded = []
    for item in ordered:
        if item['method'] != 'subworkflow':
            expanded.append(item)
            continue
        remap = {n['id']: item['id'] if n['id'] == item['root'] else 'inner_' + digest([item['id'], n['id']])[:32] for n in item['nested']}
        expanded.extend(({**copy.deepcopy(n), 'id': remap[n['id']], 'inputs': {p: remap[v] for p, v in n['inputs'].items()}} for n in item['nested']))
    if len(expanded) > max_nodes or len({n['id'] for n in expanded}) != len(expanded):
        raise ValueError('Expanded workflow exceeds node budget or has colliding identities')
    return {'ordered': expanded, 'outputs': outputs, 'types': types, 'document': copy.deepcopy(document), 'binding': copy.deepcopy(expected)}

def select_scope(compiled, scope, selected):
    nodes = {n['id']: n for n in compiled['ordered']}
    if not isinstance(selected, list) or len(set(selected)) != len(selected) or any((n not in nodes for n in selected)):
        raise ValueError('Invalid selected nodes')
    if scope == 'workflow_outputs':
        if selected:
            raise ValueError('Workflow outputs scope does not accept a node selection')
        roots = compiled['outputs']
        if not roots:
            raise ValueError('Choose at least one desired output before planning')
    elif scope == 'selected_subgraph':
        roots = selected
        if not roots:
            raise ValueError('Select nodes before planning')
    else:
        raise ValueError('Unsupported execution scope')
    closure = set(roots)
    queue = list(roots)
    for key in queue:
        for ancestor in nodes[key]['inputs'].values():
            if ancestor not in closure:
                closure.add(ancestor)
                queue.append(ancestor)
    return ([n for n in compiled['ordered'] if n['id'] in closure], roots)
