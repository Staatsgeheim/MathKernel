# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Optional compiled sparse conic search; all claims are checked in conic.py."""
from __future__ import annotations
import time
from .conic import ConicCertificate, verify_certificate
from .engineering import EngineeringResult, checked_result, numeric_array
from .optimization import _rational_candidate


def pack_problem(problem):
    """Pack PSD upper triangles in column order with sqrt(2) off diagonals.

    See https://clarabel.org/stable/api_cone_types/ . This is an approximate
    backend coordinate transform, not a rewrite of the original exact model.
    """
    import numpy as np
    import clarabel
    from scipy import sparse
    rows, cols, values = [], [], []
    for i, row in enumerate(problem.A):
        for j, value in enumerate(row):
            if value != 0:
                rows.append(i);cols.append(j);values.append(value)
    original = sparse.csr_matrix((numeric_array(values, real=True), (rows, cols)),
                                 shape=(len(problem.b), len(problem.c)))
    selected, scales, cones, mapping = [], [], [], []
    offset = 0
    for block in problem.cones:
        if block.kind == 'psd':
            cones.append(clarabel.PSDTriangleConeT(block.dimension))
            for j in range(block.dimension):
                for i in range(j+1):
                    selected.append(offset+i*block.dimension+j)
                    scale = 1. if i == j else np.sqrt(2.)
                    scales.append(scale)
                    mapping.append((offset+i*block.dimension+j, offset+j*block.dimension+i, scale))
        else:
            cones.append(clarabel.NonnegativeConeT(block.dimension) if block.kind == 'nonnegative'
                         else clarabel.SecondOrderConeT(block.dimension))
            for i in range(block.dimension):
                selected.append(offset+i);scales.append(1.);mapping.append((offset+i, offset+i, 1.))
        offset += block.size
    selected = np.asarray(selected, dtype=int);scales = np.asarray(scales)
    packed = original[selected].multiply(scales[:, None]).tocsc()
    rhs = numeric_array(problem.b, real=True)[selected]*scales
    eq = sparse.csc_matrix(np.asarray(problem.A_eq, dtype=float).reshape((len(problem.b_eq), len(problem.c))))
    if problem.b_eq:
        packed = sparse.vstack((packed, eq), format='csc')
        rhs = np.r_[rhs, numeric_array(problem.b_eq, real=True)]
        cones.append(clarabel.ZeroConeT(len(problem.b_eq)))
    if not np.all(np.isfinite(packed.data)) or not np.all(np.isfinite(rhs)):
        raise ValueError('conic packing exceeded float64 range')
    c = numeric_array(problem.c, real=True)*(1 if problem.sense == 'min' else -1)
    return packed, rhs, cones, mapping, original, eq, c


def unpack_dual(packed, mapping, rows):
    import numpy as np
    z = np.zeros(rows)
    for value, (i, j, scale) in zip(packed, mapping):
        z[i] = z[j] = value/scale
    return z


def cone_violation(values, blocks):
    import numpy as np
    offset = 0; violations = []
    for block in blocks:
        part = values[offset:offset+block.size]
        if block.kind == 'nonnegative':
            violation = max(0., float(-np.min(part)))
        elif block.kind == 'second_order':
            violation = max(0., float(np.linalg.norm(part[1:])-part[0]))
        else:
            matrix = part.reshape(block.dimension, block.dimension)
            violation = max(0., float(-np.linalg.eigvalsh(matrix)[0]), float(np.max(abs(matrix-matrix.T))))
        violations.append(violation);offset += block.size
    return max(violations, default=0.)


def _isolated_conic_solve(problem, *, max_iterations=1000, tolerance=1e-9, time_limit=30., reconstruction_denominator=1000000):
    import numpy as np
    import clarabel
    from scipy import sparse
    packed, rhs, cones, mapping, original, eq, c = pack_problem(problem)
    settings = clarabel.DefaultSettings()
    settings.verbose = False;settings.max_iter = max_iterations;settings.time_limit = time_limit
    settings.tol_gap_abs = tolerance;settings.tol_gap_rel = tolerance;settings.tol_feas = tolerance
    settings.max_threads = 1
    n = len(c);start = time.monotonic()
    def run(objective):
        settings.time_limit = max(.001, time_limit-(time.monotonic()-start))
        return clarabel.DefaultSolver(sparse.csc_matrix((n, n)), objective, packed, rhs, cones, settings).solve()
    raw = run(c)
    status = str(raw.status)
    x = np.asarray(raw.x);z = unpack_dual(raw.z, mapping, len(problem.b))
    y = np.asarray(raw.z[len(mapping):])
    if not all(np.all(np.isfinite(v)) for v in (x, z, y)):
        return EngineeringResult(operation='solve', status='unknown', trust='unknown',
            details={'conclusion': 'nonfinite_solver_candidate', 'solver_status': status})
    b = numeric_array(problem.b, real=True);f = numeric_array(problem.b_eq, real=True)
    slack = b-original@x
    residual = {'primal_cone': cone_violation(slack, problem.cones),
        'primal_equalities': float(np.max(abs(eq@x-f), initial=0)),
        'dual_cone': cone_violation(z, problem.cones),
        'stationarity': float(np.max(abs(c+original.T@z+eq.T@y), initial=0)),
        'duality_gap': float(c@x+b@z+f@y)}
    details = {'solver': 'clarabel', 'solver_version': clarabel.__version__, 'solver_status': status,
        'solver_iterations': int(raw.iterations), 'primal': x.tolist(), 'residuals': residual,
        'tolerance': tolerance, 'global_optimality_certified': False, 'accepted': False,
        'conclusion': 'numerical_candidate', 'original_rows': len(b)+len(f), 'packed_rows': packed.shape[0],
        'packed_nonzeros': packed.nnz, 'psd_backend_coordinates': 'upper_column_svec_sqrt2', 'max_threads': 1}
    reported_outcome = status in {'PrimalInfeasible', 'AlmostPrimalInfeasible', 'DualInfeasible', 'AlmostDualInfeasible'}
    report = checked_result('solve', float(c@x)*(1 if problem.sense == 'min' else -1),
        method='clarabel_sparse_product_cone', trust='numeric', precision=53, candidate=True,
        residual=residual, details=details)
    # A dual-infeasible solver vector is only a recession candidate. A separately
    # searched feasible anchor is also necessary to prove primal unboundedness.
    anchor = None
    if status in {'DualInfeasible', 'AlmostDualInfeasible'} and time.monotonic()-start < time_limit:
        feasibility = run(np.zeros(n))
        anchor = np.asarray(feasibility.x)
        if not np.all(np.isfinite(anchor)):
            anchor = None
    certificate = None
    for denominator in sorted({min(reconstruction_denominator, d) for d in (100, 10000, reconstruction_denominator)}):
        if status in {'PrimalInfeasible', 'AlmostPrimalInfeasible'}:
            certificate = ConicCertificate(kind='infeasible', cone_dual=_rational_candidate(z, denominator),
                                          equality_dual=_rational_candidate(y, denominator))
        elif status in {'DualInfeasible', 'AlmostDualInfeasible'}:
            if anchor is None:
                continue
            certificate = ConicCertificate(kind='unbounded', primal=_rational_candidate(anchor, denominator),
                                          ray=_rational_candidate(x, denominator))
        else:
            certificate = ConicCertificate(primal=_rational_candidate(x, denominator),
                cone_dual=_rational_candidate(z, denominator), equality_dual=_rational_candidate(y, denominator))
        result = verify_certificate(problem, certificate)
        if result.details['accepted']:
            result.operation = 'solve'
            result.details.update(search_diagnostics=details, reconstruction_max_denominator=denominator)
            diagnostic = report.claim_evidence['solve']
            for item in (*diagnostic.computation, *diagnostic.numerical):
                item.role = 'diagnostic'
            bundle = result.claim_evidence['verify_certificate']
            bundle.computation.extend(diagnostic.computation);bundle.numerical.extend(diagnostic.numerical)
            return result, certificate
    if reported_outcome:
        report.status = 'unknown';report.trust = 'unknown';report.value = None
        report.details['conclusion'] = 'unverified_solver_report'
    if certificate is not None:
        report.details['certificate'] = certificate.model_dump(mode='json')
        return report, certificate
    return report


def solve(problem, *, max_iterations=1000, tolerance=1e-9, time_limit=30., reconstruction_denominator=1000000):
    from .engines import run_in_subprocess
    output = run_in_subprocess(_isolated_conic_solve, time_limit, problem,
        max_iterations=max_iterations, tolerance=tolerance,
        time_limit=time_limit*.9, reconstruction_denominator=reconstruction_denominator)
    result = output[0] if isinstance(output, tuple) else output
    result.details['process_isolation'] = 'fresh_interpreter_hard_killable'
    return output
