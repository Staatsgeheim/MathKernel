# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Checked continuous/discrete LQR and steady-state predictor construction.

Numerical ARE search is a candidate generator. Exact certificates use original
rational data, positivity, the defining equation and independent stability.
"""
from __future__ import annotations
import sympy as sp
from pydantic import model_validator
from .engineering import EngineeringModel, checked_result, validate_scalars, arithmetic_trust, cap_trust, numeric_array, sympy_samples
from .control_systems import _matrix_tuple, _system_trust, stability
from .control_design import _copy_system, channel_units
from .optimization import exact_psd, _rational_candidate


class RiccatiCertificate(EngineeringModel):
    P: tuple[tuple[sp.Expr, ...], ...]

    @model_validator(mode='after')
    def validate_matrix(self):
        n=len(self.P)
        if not n or any(len(row)!=n for row in self.P):
            raise ValueError('P must be nonempty and square')
        validate_scalars((v for row in self.P for v in row),real=True)
        if any(v.free_symbols for row in self.P for v in row):
            raise ValueError('P must be concrete')
        if any(self.P[i][j]!=self.P[j][i] for i in range(n) for j in range(i)):
            raise ValueError('P must be explicitly symmetric')
        return self


def _pd(M):
    # A full set of positive Schur pivots proves PD without recomputing minors.
    if M != M.T: return False
    psd, pivots = exact_psd(M)
    return psd is True and len(pivots) == M.rows


def _data(system,Q,R,kalman):
    A,B,C,D=(sp.Matrix(v) for v in (system.A,system.B,system.C,system.D))
    a,b=(A.T,C.T) if kalman else (A,B)
    q,r=sp.Matrix(Q),sp.Matrix(R)
    if q.shape!=a.shape or r.shape!=(b.cols,b.cols):
        raise ValueError('weight/covariance dimensions must match states and control inputs or measurements')
    if q!=q.T or r!=r.T:
        raise ValueError('weights/covariances must be explicitly symmetric')
    entries=tuple(v for m in (A,B,C,D,q,r) for v in m)
    validate_scalars(entries,real=True)
    if any(v.free_symbols for v in entries):
        raise ValueError('Riccati synthesis requires concrete real data')
    return a,b,q,r


def _closed(system,P,K,trust,kalman):
    A,B,C,D=(sp.Matrix(v) for v in (system.A,system.B,system.C,system.D))
    if not kalman:
        return {'output':_copy_system(system,A-B*K,B,C-D*K,D,input_trust=trust)}
    L=K.T;inputs,outputs=channel_units(system);states=system.state_units or ('',)*A.rows
    return {'output':_copy_system(system,A-L*C,(B-L*D).row_join(L),sp.eye(A.rows),sp.zeros(A.rows,B.cols+C.rows),
                input_units=(*inputs,*outputs),output_units=states,input_unit='',output_unit='',input_trust=trust),
            'error':_copy_system(system,A-L*C,sp.zeros(A.rows,B.cols),sp.eye(A.rows),sp.zeros(A.rows,B.cols),
                output_units=states,output_unit='',input_trust=trust)}


def _semantics(kalman,discrete):
    if kalman:
        return {'claim_scope':'stabilizing covariance equation and estimator algebra; no empirical or Gaussian optimality claim',
            'noise_assumptions':['zero-mean white process and measurement noise','no process/measurement cross-covariance',
                'W acts directly in state coordinates','known deterministic control input'],
            'covariance_kind':'prior prediction covariance' if discrete else 'continuous error covariance',
            'estimator_equation':'xhat_next=A*xhat+B*u+L*(y-C*xhat-D*u)' if discrete else
                                 'dxhat=A*xhat+B*u+L*(dy-(C*xhat+D*u)*dt)',
            'noise_units':'per sample covariances' if discrete else 'continuous spectral intensities',
            'gaussianity_verified':False,'model_validated':False}
    return {'claim_scope':'infinite-horizon LQR among controls with terminal x.T*P*x tending to zero',
        'cost':'sum(x.T*Q*x+u.T*R*u)' if discrete else 'integral(x.T*Q*x+u.T*R*u) dt',
        'control_law':'u=-K*x','value_function':'x0.T*P*x0',
        'cross_weight':'zero','unrestricted_finite_cost_optimality_claimed':False}


def verify(system,Q,R,certificate,*,kalman=False,input_trust='exact',certificate_trust='exact'):
    a,b,q,r=_data(system,Q,R,kalman);P=sp.Matrix(certificate.P)
    if P.shape!=a.shape:raise ValueError('P dimension must equal state dimension')
    discrete=system.time_domain=='discrete'
    trust=cap_trust(_system_trust(system),input_trust,certificate_trust,arithmetic_trust((*q,*r,*P)))
    checks={'Q_psd':exact_psd(q)[0] is True,'R_pd':_pd(r),'P_psd':exact_psd(P)[0] is True}
    S=r+b.T*P*b if discrete else r
    checks['gain_denominator_pd']=_pd(S) if discrete else checks['R_pd']
    K=None;residual=None;stable=False
    if checks['gain_denominator_pd']:
        rhs=b.T*P*a if discrete else b.T*P
        K=S.solve(rhs,method='GJ')
        checks['gain_identity']=all(sp.cancel(v)==0 for v in S*K-rhs)
        residual=a.T*P*a-P+q-a.T*P*b*K if discrete else a.T*P+P*a+q-P*b*K
        checks['riccati_identity']=all(sp.cancel(v)==0 for v in residual)
        closed=a-b*K
        # Only A determines this independent exact stability criterion.
        probe=_copy_system(system,closed,sp.zeros(a.rows,1),sp.zeros(1,a.rows),sp.zeros(1,1),
            input_unit='',output_unit='',input_units=(),output_units=(),state_units=(),input_trust=trust)
        stability_result=stability(probe);stable=stability_result.value is True
        checks['strict_stability']=stable
    else:checks.update(riccati_identity=False,strict_stability=False)
    rational=all(v.is_Rational for v in (*a,*b,*q,*r,*P))
    accepted=all(checks.values()) and trust=='exact' and rational
    name='kalman' if kalman else 'lqr'
    result=checked_result(name,None if K is None else _matrix_tuple(K.T if kalman else K),
        method='original_data_stabilizing_riccati',trust=trust,
        checks={'certificate_accepted':True} if accepted else {},candidate=not accepted,
        witness=certificate.model_dump(mode='json'),details={'accepted':accepted,
            'conclusion':('certified_stabilizing_covariance_equation' if kalman else 'certified_stabilizing_lqr') if accepted else 'uncertified_candidate',
            'certificate_checks':checks,'P':certificate.P,'gain':None if K is None else _matrix_tuple(K.T if kalman else K),
            'riccati_residual':None if residual is None else residual.tolist(),
            'certificate':certificate.model_dump(mode='json'),**_semantics(kalman,discrete)})
    outputs={'certificate':certificate}
    if accepted:outputs.update(_closed(system,P,K,trust,kalman))
    return result,outputs


def _isolated_riccati_synthesize(system,Q,R,*,kalman=False,mode='exact',certificate=None,input_trust='exact',certificate_trust='exact',
               tolerance=1e-9,max_exact_order=8,reconstruction_denominator=1000000):
    if certificate is not None:
        if len(system.A)>max_exact_order:
            raise ValueError('exact Riccati verification exceeds max_exact_control_order')
        return verify(system,Q,R,certificate,kalman=kalman,input_trust=input_trust,certificate_trust=certificate_trust)
    if mode!='numeric':
        raise NotImplementedError('supply a Riccati certificate for exact mode or explicitly choose numeric search')
    import numpy as np
    from scipy.linalg import solve_continuous_are,solve_discrete_are,cho_factor,cho_solve
    a,b,q,r=_data(system,Q,R,kalman)
    def arr(M):return numeric_array(tuple(M),real=True).reshape(M.shape)
    A,B,Qn,R=map(arr,(a,b,q,r));discrete=system.time_domain=='discrete'
    # Numerical checks do not certify semidefiniteness. Original rational data
    # are independently tested after reconstruction before any exact promotion.
    if np.min(np.linalg.eigvalsh(Qn)) < -tolerance or np.min(np.linalg.eigvalsh(R))<=0:
        raise ValueError('Q/W must be PSD and R/V positive definite')
    P=(solve_discrete_are if discrete else solve_continuous_are)(A,B,Qn,R)
    P=(P+P.T)/2
    S=R+B.T@P@B if discrete else R
    rhs=B.T@P@A if discrete else B.T@P
    K=cho_solve(cho_factor(S,lower=True),rhs)
    closed=A-B@K
    residual=A.T@P@A-P+Qn-A.T@P@B@K if discrete else A.T@P+P@A+Qn-P@B@K
    eig=np.linalg.eigvals(closed)
    margin=float(1-np.max(abs(eig))) if discrete else float(-np.max(eig.real))
    scale=max(1.,np.linalg.norm(P,ord=np.inf),np.linalg.norm(Qn,ord=np.inf),np.linalg.norm(A,ord=np.inf)**2*np.linalg.norm(P,ord=np.inf),np.linalg.norm(rhs,ord=np.inf)*np.linalg.norm(K,ord=np.inf))
    if not np.isfinite(scale) or not np.isfinite(margin):
        raise ValueError('Riccati diagnostic scaling exceeded float64 range')
    relative=float(np.linalg.norm(residual,ord=np.inf)/scale)
    gain_residual=float(np.linalg.norm(S@K-rhs,ord=np.inf)/max(1.,np.linalg.norm(rhs,ord=np.inf)))
    checks={'gain_residual':gain_residual<=tolerance,'riccati_residual':relative<=tolerance,'P_psd':bool(np.min(np.linalg.eigvalsh(P))>=-tolerance),
            'strict_stability':margin>tolerance}
    if not all(np.all(np.isfinite(v)) for v in (P,K,residual)):
        raise ValueError('Riccati solver produced nonfinite output')
    name='kalman' if kalman else 'lqr';trust=cap_trust(_system_trust(system),input_trust,'numeric')
    details={'accepted':False,'conclusion':'numerical_stabilizing_candidate' if all(checks.values()) else 'numerical_candidate',
        'numeric_checks':checks,'relative_residual':relative,'gain_relative_residual':gain_residual,'residual_scale':float(scale),'tolerance':tolerance,
        'stability_margin':margin,'stability_margin_is_rigorous':False,'solver':'scipy_dare' if discrete else 'scipy_care',
        **_semantics(kalman,discrete)}
    report=checked_result(name,K.T.tolist() if kalman else K.tolist(),method=details['solver'],trust=trust,candidate=True,
        residual=relative,precision=53,details=details)
    if a.rows<=max_exact_order and input_trust=='exact' and _system_trust(system)=='exact':
        for denominator in sorted({100,min(10000,reconstruction_denominator),reconstruction_denominator}):
            denominator=min(denominator,reconstruction_denominator)
            values=_rational_candidate(P.ravel(),denominator)
            cert=RiccatiCertificate(P=_matrix_tuple(sp.Matrix(a.rows,a.rows,values)))
            result,outputs=verify(system,Q,R=_matrix_tuple(r),certificate=cert,kalman=kalman,input_trust=input_trust)
            if result.details['accepted']:
                result.details.update(search_diagnostics=details,reconstruction_max_denominator=denominator)
                diagnostic=report.claim_evidence[name]
                for item in (*diagnostic.computation,*diagnostic.numerical):item.role='diagnostic'
                result.claim_evidence[name].computation.extend(diagnostic.computation)
                result.claim_evidence[name].numerical.extend(diagnostic.numerical)
                return result,outputs
    pm=sp.Matrix(a.rows,a.rows,sympy_samples(P.ravel()));km=sp.Matrix(K.shape[0],K.shape[1],sympy_samples(K.ravel()))
    cert=RiccatiCertificate(P=_matrix_tuple(pm));outputs={'certificate':cert}
    if all(checks.values()):outputs.update(_closed(system,pm,km,trust,kalman))
    report.details.update(P=cert.P,gain=_matrix_tuple(km.T if kalman else km),certificate=cert.model_dump(mode='json'))
    return report,outputs


def synthesize(system,Q,R,*,kalman=False,mode='exact',certificate=None,input_trust='exact',certificate_trust='exact',
               tolerance=1e-9,max_exact_order=8,reconstruction_denominator=1000000,time_limit=30.0,
               _isolate=True):
    options = dict(kalman=kalman, mode=mode, certificate=certificate, input_trust=input_trust,
        certificate_trust=certificate_trust, tolerance=tolerance, max_exact_order=max_exact_order,
        reconstruction_denominator=reconstruction_denominator)
    if certificate is not None or mode != 'numeric' or not _isolate:
        return _isolated_riccati_synthesize(system,Q,R,**options)
    from .engines import run_in_subprocess
    output = run_in_subprocess(_isolated_riccati_synthesize, time_limit, system,Q,R,**options)
    output[0].details['process_isolation'] = 'fresh_interpreter_hard_killable'
    return output
