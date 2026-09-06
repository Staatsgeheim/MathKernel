# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""MIMO operations, sampled-system conversion and checked controller construction."""
from __future__ import annotations
import sympy as sp
from pydantic import model_validator
from .engineering import (EngineeringModel,arithmetic_trust,cap_trust,checked_result,
                          exact_zero,numeric_array,sympy_samples,validate_scalars)
from .control_systems import (TransferFunction,StateSpaceSystem,_matrix_tuple,
                              _system_trust,polynomial)
from .units import parse_unit


class TransferMatrix(EngineeringModel):
    entries: tuple[tuple[TransferFunction,...],...]
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_matrix(self):
        if not self.entries or not self.entries[0] or any(len(row)!=len(self.entries[0]) for row in self.entries):
            raise ValueError("transfer matrix must be nonempty and rectangular")
        first=self.entries[0][0]
        if any((v.time_domain,v.sample_time)!=(first.time_domain,first.sample_time) for row in self.entries for v in row):
            raise ValueError("transfer matrix entries require a common time base")
        cap_trust(self.input_trust)
        return self


def channel_units(system):
    return (system.input_units or (system.input_unit,)*len(system.B[0]),
            system.output_units or (system.output_unit,)*len(system.C))


def coefficient_units(system):
    inputs,outputs=channel_units(system)
    if not system.state_units:
        raise ValueError("coefficient-unit inference requires explicitly declared state_units")
    states=[parse_unit(v) for v in system.state_units]
    u,y=[parse_unit(v) for v in inputs],[parse_unit(v) for v in outputs]
    def ratio(top,bottom,derivative=False):
        dimension=top.dimension/bottom.dimension
        if derivative and system.time_domain=="continuous":
            dimension=dimension/parse_unit("s").dimension
        return {"dimension":str(dimension),"SI_scale":str(top.scale/bottom.scale)}
    value={"A":[[ratio(a,b,True) for b in states] for a in states],
           "B":[[ratio(a,b,True) for b in u] for a in states],
           "C":[[ratio(a,b) for b in states] for a in y],
           "D":[[ratio(a,b) for b in u] for a in y]}
    return checked_result("coefficient_units",value,method="state_coordinate_dimensional_ratios",
        trust=_system_trust(system),details={"interpretation":"required units of coefficient magnitudes",
        "time_unit":"s","supplied_coefficient_units_checked":False})


def _copy_system(system,A,B,C,D,**updates):
    data=system.model_dump(mode="python")
    data.update(A=_matrix_tuple(A),B=_matrix_tuple(B),C=_matrix_tuple(C),D=_matrix_tuple(D))
    data.update(updates)
    return StateSpaceSystem(**data)


def discretize(system,dt,*,method="zoh",mode="exact",max_exact_order=8):
    if system.time_domain!="continuous":
        raise ValueError("discretize requires a continuous system")
    validate_scalars((dt,),real=True)
    if dt.is_positive is not True or dt.free_symbols:
        raise ValueError("sample_time must be a concrete positive value in seconds")
    if method not in {"zoh","bilinear"} or mode not in {"exact","numeric"}:
        raise ValueError("method must be zoh or bilinear; mode must be exact or numeric")
    A,B,C,D=(sp.Matrix(m) for m in (system.A,system.B,system.C,system.D))
    n,m=A.rows,B.cols
    trust=cap_trust(_system_trust(system),arithmetic_trust((dt,)))
    residual=None;budget=None;witness=None
    if mode=="numeric":
        import numpy as np
        from scipy.linalg import expm,solve
        def arr(mat):return numeric_array(tuple(mat)).reshape(mat.shape)
        a,b,c,d=map(arr,(A,B,C,D));h=float(dt)
        if method=="zoh":
            augmented=np.zeros((n+m,n+m),dtype=complex)
            augmented[:n,:n]=a;augmented[:n,n:]=b
            full=expm(h*augmented);half=expm(h*augmented/2)
            residual=float(np.max(abs(full-half@half)))
            ad,bd,cd,dd=full[:n,:n],full[:n,n:],c,d
            budget=float(512*np.finfo(float).eps*max(1,np.max(abs(full)),np.max(abs(half))**2)*(n+m))
            checks={"semigroup_consistency":bool(residual<=budget)}
        else:
            R=np.eye(n)-h*a/2
            try:
                solved=solve(R,np.column_stack((np.eye(n)+h*a/2,h*b,np.eye(n))))
            except np.linalg.LinAlgError as exc:
                raise ValueError("singular bilinear discretization") from exc
            ad,bd=solved[:,:n],solved[:,n:n+m]
            inverse_action=solved[:,n+m:]
            cd=c@inverse_action;dd=d+cd@b*h/2
            residual=float(np.max(abs(R@solved-np.column_stack((np.eye(n)+h*a/2,h*b,np.eye(n))))))
            budget=float(512*np.finfo(float).eps*max(1,np.max(abs(R))*np.max(abs(solved)))*n)
            checks={"linear_solve_residual":bool(residual<=budget)}
        def mat(values):return sp.Matrix(values.shape[0],values.shape[1],sympy_samples(values.ravel()))
        Ad,Bd,Cd,Dd=map(mat,(ad,bd,cd,dd));trust=cap_trust(trust,"numeric")
    else:
        if n>max_exact_order:
            raise ValueError(f"exact discretization exceeds max_exact_control_order={max_exact_order}")
        if any(v.free_symbols for mat in (A,B,C,D) for v in mat):
            raise NotImplementedError("parametric discretization requires case-specific invertibility assumptions")
        if method=="bilinear":
            R=sp.eye(n)-dt*A/2
            if R.det()==0:
                raise ValueError("singular bilinear discretization")
            solved=R.solve((sp.eye(n)+dt*A/2).row_join(dt*B).row_join(sp.eye(n)),method="GJ")
            Ad,Bd=solved[:,:n],solved[:,n:n+m]
            Cd=(C*solved[:,n+m:]).applyfunc(sp.cancel);Dd=(D+Cd*B*dt/2).applyfunc(sp.cancel)
            checks={"solve_identity":all(exact_zero(x) is True for x in R*solved-(sp.eye(n)+dt*A/2).row_join(dt*B).row_join(sp.eye(n)))}
            witness={"left_matrix":R.tolist(),"solution":solved.tolist()}
        else:
            M=A.row_join(B).col_join(sp.zeros(m,n+m))
            # Nilpotent systems have a finite exact polynomial exponential.
            power=sp.eye(n+m);E=power.copy();nilpotent=False
            for degree in range(1,n+m+1):
                power=power*M
                if power.is_zero_matrix:
                    nilpotent=True;break
                E+=dt**degree*power/sp.factorial(degree)
            if nilpotent:
                Ad,Bd,Cd,Dd=E[:n,:n],E[:n,n:],C,D
                checks={"nilpotent_exponential_termination":True}
                witness={"nilpotence_degree":degree,"zero_power":power.tolist()}
            else:
                t=sp.Dummy("tau",real=True)
                trajectory=(t*M).exp()
                checks={"exponential_differential_identity":all(exact_zero(x) is True for x in trajectory.diff(t)-M*trajectory),
                        "identity_initial_condition":trajectory.subs(t,0)==sp.eye(n+m)}
                E=trajectory.subs(t,dt)
                Ad,Bd,Cd,Dd=E[:n,:n],E[:n,n:],C,D
                trust=cap_trust(trust,"symbolic")
                witness={"differential_equation":"E'=M E; E(0)=I"}
    derived=_copy_system(system,Ad,Bd,Cd,Dd,time_domain="discrete",sample_time=dt,input_trust=trust)
    return checked_result("discretize",derived,method=f"{mode}_{method}_state_conversion",trust=trust,
        checks=checks,witness=witness,residual=residual,precision=53 if mode=="numeric" else None,
        details={"method":method,"sample_time":dt,"verification_absolute_tolerance":budget,
                 "meaning":"exact sampling model for piecewise-constant inputs" if method=="zoh" else "bilinear transfer substitution; not exact sampling",
                 "continuous_trajectory_certified":False}),derived


def state_frequency_response(system,frequencies,*,max_work=1_000_000):
    import numpy as np
    A,B,C,D=(sp.Matrix(m) for m in (system.A,system.B,system.C,system.D))
    if len(frequencies)*C.rows*B.cols>max_work:
        raise ValueError("MIMO frequency output exceeds work budget")
    def arr(m):return numeric_array(tuple(m)).reshape(m.shape)
    a,b,c,d=map(arr,(A,B,C,D));omega=numeric_array(frequencies,real=True)
    if np.any(omega<0):raise ValueError("frequencies must be nonnegative rad/s")
    if system.time_domain=="discrete":
        dt=float(system.sample_time)
        if np.any(omega>np.pi/dt):raise ValueError("frequency exceeds Nyquist")
        points=np.exp(1j*omega*dt)
    else:points=1j*omega
    values=[];residual=0.;scale=1.
    # Bounded batched solves; avoid allocating frequency_count*n*n at once.
    batch=max(1,min(64,max_work//max(1,A.rows*A.rows)))
    for offset in range(0,len(points),batch):
        matrices=points[offset:offset+batch,None,None]*np.eye(A.rows)-a
        try:solutions=np.linalg.solve(matrices,np.broadcast_to(b,(len(matrices),*b.shape)))
        except np.linalg.LinAlgError as exc:raise ValueError("frequency resolvent is singular") from exc
        response=c@solutions+d
        residual=max(residual,float(np.max(abs(matrices@solutions-b))))
        scale=max(scale,float(np.max(abs(matrices)))*float(np.max(abs(solutions))))
        values.extend(tuple(tuple(sympy_samples(row)) for row in matrix) for matrix in response)
    tolerance=256*np.finfo(float).eps*scale*A.rows
    return checked_result("frequency_response",tuple(values),method="batched_mimo_resolvent_solves",
        trust=cap_trust(_system_trust(system),"numeric"),checks={"resolvent_residual":bool(residual<=tolerance)},
        residual=residual,precision=53,details={"shape":[len(frequencies),C.rows,B.cols],"angular_frequencies":frequencies,
            "frequency_unit":"rad/s","verification_absolute_tolerance":float(tolerance),"batch_size":batch})


def state_feedback(system,gain):
    A,B,C,D=(sp.Matrix(m) for m in (system.A,system.B,system.C,system.D));K=sp.Matrix(gain)
    if K.shape!=(B.cols,A.rows):raise ValueError("feedback gain must be inputs by states")
    trust=cap_trust(_system_trust(system),arithmetic_trust(tuple(K)))
    derived=_copy_system(system,A-B*K,B,C-D*K,D,input_trust=trust)
    return checked_result("state_feedback",derived,method="closed_loop_substitution",trust=trust,
        checks={"state_equation":True,"feedthrough_output_equation":True},
        witness={"gain":K.tolist(),"control_law":"u=r-K*x"},
        details={"stability_claimed":False}),derived


def _isolated_place_poles(a,b,poles,max_iterations,rtol):
    import numpy as np
    from scipy.signal import place_poles as scipy_place
    raw=scipy_place(a,b,poles,method="YT",rtol=rtol,maxiter=max_iterations)
    return {"gain":raw.gain_matrix,
            "iterations":int(raw.nb_iter) if np.isfinite(raw.nb_iter) else None,
            "rtol":float(raw.rtol) if np.isfinite(raw.rtol) else None}


def place_poles(system,poles,*,mode="exact",observer=False,max_iterations=1000,rtol=1e-3,time_limit=30.0):
    A,B,C,D=(sp.Matrix(m) for m in (system.A,system.B,system.C,system.D))
    n=A.rows
    if len(poles)!=n:raise ValueError("pole count must equal state dimension")
    validate_scalars(poles)
    if any(v.free_symbols for v in (*poles,*A,*B,*C,*D)):
        raise NotImplementedError("pole placement currently requires concrete coefficients and poles")
    z=sp.Dummy("pole");desired=sp.Poly(sp.prod(z-p for p in poles),z)
    if any(sp.simplify(sp.im(v))!=0 for v in desired.all_coeffs()):
        raise ValueError("real controller construction requires conjugate-paired complex poles")
    a,b=(A.T,C.T) if observer else (A,B)
    trust=cap_trust(_system_trust(system),arithmetic_trust(poles))
    if mode=="exact":
        if b.cols!=1:raise NotImplementedError("exact pole placement supports single-input or single-output dual systems; choose numeric mode for MIMO")
        blocks=[];block=b
        for index in range(n):
            blocks.append(block)
            if index + 1 < n:
                block=a*block
        controllability=sp.Matrix.hstack(*blocks)
        if controllability.det()==0:raise ValueError("system is not controllable/observable for requested pole placement")
        e=sp.zeros(n,1);e[-1]=1
        row=controllability.T.solve(e,method="GJ").T
        phi=sp.zeros(n)
        for coefficient in desired.all_coeffs():phi=phi*a+coefficient*sp.eye(n)
        gain=(row*phi).applyfunc(sp.cancel)
        err=a-b*gain
        checks={"characteristic_polynomial":all(exact_zero(x-y) is True for x,y in zip(err.charpoly(z).all_coeffs(),desired.all_coeffs()))}
        residual=None;details={"method":"ackermann_dual" if observer else "ackermann","requested_poles":poles}
    elif mode=="numeric":
        import numpy as np
        from .engines import run_in_subprocess
        def arr(m):return numeric_array(tuple(m),real=True).reshape(m.shape)
        raw=run_in_subprocess(_isolated_place_poles,time_limit,arr(a),arr(b),numeric_array(poles),max_iterations,rtol)
        raw_gain=raw["gain"]
        gain=sp.Matrix(raw_gain.shape[0],raw_gain.shape[1],sympy_samples(raw_gain.ravel()))
        actual=np.poly(arr(a)-arr(b)@raw_gain)
        target=np.poly(numeric_array(poles))
        residual=float(np.max(abs(actual-target)));threshold=float(1e-7*max(1,np.max(abs(target))))
        checks={"characteristic_polynomial_numeric":bool(residual<=threshold)}
        trust=cap_trust(trust,"numeric")
        details={"method":"scipy_YT","requested_poles":poles,"iterations":raw["iterations"],
                 "solver_rtol":raw["rtol"], "requested_rtol":rtol,"max_iterations":max_iterations,
                 "verification_absolute_tolerance":threshold,"process_isolation":"fresh_interpreter_hard_killable"}
    else:raise ValueError("mode must be exact or numeric")
    if observer:
        L=gain.T
        inputs,outputs=channel_units(system)
        states=system.state_units or ("",)*n
        observed=_copy_system(system,A-L*C,(B-L*D).row_join(L),sp.eye(n),sp.zeros(n,B.cols+C.rows),
            input_trust=trust,input_units=(*inputs,*outputs),output_units=states,input_unit="",output_unit="")
        error=_copy_system(system,A-L*C,sp.zeros(n,B.cols),sp.eye(n),sp.zeros(n,B.cols),
            input_trust=trust,output_units=states,output_unit="")
        gain=L;derived={"output":observed,"error":error}
        details["observer_equation"]="xhat'=(A-LC)xhat+(B-LD)u+L*y"
    else:
        _,closed=state_feedback(system,_matrix_tuple(gain))
        closed=closed.model_copy(update={"input_trust":trust});derived=closed
        details["control_law"]="u=r-K*x"
    details["gain"]=_matrix_tuple(gain);details["stability_claimed"]=False
    result=checked_result("observer" if observer else "place_poles",_matrix_tuple(gain),
        method=details["method"]+"_coefficient_check",trust=trust,checks=checks,
        witness={"gain":_matrix_tuple(gain),"requested_polynomial":desired.all_coeffs()},
        residual=residual,precision=53 if mode=="numeric" else None,details=details)
    return result,derived
