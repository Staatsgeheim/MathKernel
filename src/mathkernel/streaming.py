# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
"""Immutable filter continuation states and independently checked chunk boundaries."""
from __future__ import annotations
from typing import Literal
import sympy as sp
from pydantic import StrictInt, model_validator
from .engineering import (EngineeringModel, arithmetic_trust, cap_trust, checked_result,
                          numeric_array, sympy_samples, validate_scalars, exact_zero)
from .signal_processing import DiscreteSignal, Filter
from .units import parse_unit


class FilterState(EngineeringModel):
    filter_id: str
    backend: Literal["exact_df2t", "numeric_df2t", "numeric_sos"]
    delay: tuple[sp.Expr,...]
    input_history: tuple[sp.Expr,...]
    output_history: tuple[sp.Expr,...]
    next_start: sp.Expr
    unit: str = ""
    samples_processed: StrictInt = 0
    input_trust: str = "exact"

    @model_validator(mode="after")
    def validate_state(self):
        validate_scalars((*self.delay,*self.input_history,*self.output_history,self.next_start))
        if self.next_start.is_real is not True or self.samples_processed<0:
            raise ValueError("invalid stream time/count")
        parse_unit(self.unit);cap_trust(self.input_trust)
        return self


def initial_state(filt, filter_id, *, mode="exact", start=sp.S.Zero, unit=""):
    if mode not in {"exact","numeric"}:
        raise ValueError("mode must be exact or numeric")
    if mode=="exact" and filt.sos:
        raise ValueError("SOS designs require numeric streaming mode")
    backend="exact_df2t" if mode=="exact" else "numeric_sos" if filt.sos else "numeric_df2t"
    length=2*len(filt.sos) if backend=="numeric_sos" else max(len(filt.numerator),len(filt.denominator))-1
    trust=arithmetic_trust((*filt.numerator,*filt.denominator,filt.sample_rate,start),filt.input_trust)
    if mode=="numeric":
        trust=cap_trust(trust,"numeric")
    state=FilterState(filter_id=filter_id,backend=backend,delay=(sp.S.Zero,)*length,
        input_history=(sp.S.Zero,)*(len(filt.numerator)-1),output_history=(sp.S.Zero,)*(len(filt.denominator)-1),
        next_start=start,unit=unit,input_trust=trust)
    return checked_result("initial_state",state,method="zero_filter_state",trust=trust,
        details={"backend":backend,"initial_condition":"zero_prehistory"}),state


def process(state, filt, signal, *, max_work=1_000_000):
    if sp.simplify(signal.sample_rate-filt.sample_rate)!=0:
        raise ValueError("stream sample rate differs from filter")
    if exact_zero(signal.start-state.next_start) is not True:
        raise ValueError("chunk is not contiguous with next_start; supply exact rational timestamps")
    if parse_unit(signal.unit)!=parse_unit(state.unit):
        raise ValueError("stream amplitude unit changed")
    a,b,x=filt.denominator,filt.numerator,signal.samples
    order=max(len(a),len(b))-1
    expected=2*len(filt.sos) if state.backend=="numeric_sos" else order
    if len(state.delay)!=expected or len(state.input_history)!=len(b)-1 or len(state.output_history)!=len(a)-1:
        raise ValueError("state dimensions do not match filter")
    if len(x)*(len(a)+len(b))>max_work:
        raise ValueError("stream recurrence verification exceeds work budget")
    trust=cap_trust(arithmetic_trust((*a,*b,filt.sample_rate),filt.input_trust),state.input_trust,
                    arithmetic_trust((*x,signal.start,signal.sample_rate),signal.input_trust))
    if state.backend=="exact_df2t":
        an=[sp.cancel(v/a[0]) for v in a]+[sp.S.Zero]*(order+1-len(a))
        bn=[sp.cancel(v/a[0]) for v in b]+[sp.S.Zero]*(order+1-len(b))
        delay=list(state.delay);output=[]
        for sample in x:
            y=sp.cancel(bn[0]*sample+(delay[0] if order else 0));output.append(y)
            delay=[sp.cancel((delay[j+1] if j+1<order else 0)+bn[j+1]*sample-an[j+1]*y) for j in range(order)]
        values=tuple(output);new_delay=tuple(delay)
        xx=(*state.input_history,*x);yy=(*state.output_history,*values)
        residuals=[]
        for i in range(len(x)):
            lhs=sum(a[j]*yy[len(a)-1+i-j] for j in range(len(a)))
            rhs=sum(b[j]*xx[len(b)-1+i-j] for j in range(len(b)))
            residuals.append(exact_zero(lhs-rhs))
        verdict=False if False in residuals else True if all(v is True for v in residuals) else None
        checks={"chunk_difference_equation":verdict};residual=None;budget=None
    else:
        import numpy as np
        from scipy.signal import lfilter,sosfilt
        xn,an,bn=numeric_array(x),numeric_array(a),numeric_array(b)
        if state.backend=="numeric_sos":
            sos=numeric_array(tuple(v for row in filt.sos for v in row)).reshape((-1,6))
            out,zf=sosfilt(sos,xn,zi=numeric_array(state.delay).reshape((-1,2)))
        else:
            out,zf=lfilter(bn,an,xn,zi=numeric_array(state.delay))
        values=sympy_samples(out);new_delay=sympy_samples(zf.ravel())
        xx=np.concatenate((numeric_array(state.input_history),xn))
        yy=np.concatenate((numeric_array(state.output_history),out))
        lhs=np.convolve(an,yy)[len(a)-1:len(a)-1+len(x)]
        rhs=np.convolve(bn,xx)[len(b)-1:len(b)-1+len(x)]
        residual=float(np.max(abs(lhs-rhs)))
        budget=float(256*np.finfo(float).eps*max(1,np.max(abs(lhs)),np.max(abs(rhs))))
        checks={"chunk_difference_equation_numeric":bool(residual<=budget)}
        trust=cap_trust(trust,"numeric")
    next_state=FilterState(filter_id=state.filter_id,backend=state.backend,delay=new_delay,
        input_history=(*state.input_history,*x)[-(len(b)-1):] if len(b)>1 else (),
        output_history=(*state.output_history,*values)[-(len(a)-1):] if len(a)>1 else (),
        next_start=signal.start+sp.Rational(len(x))/signal.sample_rate,unit=state.unit,
        samples_processed=state.samples_processed+len(x),input_trust=trust)
    output=signal.model_copy(update={"samples":values,"input_trust":trust})
    result=checked_result("process",values,method=state.backend+"_stream",trust=trust,checks=checks,
        residual=residual,precision=53 if state.backend!="exact_df2t" else None,
        details={"backend":state.backend,"next_start":next_state.next_start,"samples_processed":next_state.samples_processed,
                 "verification_absolute_tolerance":budget,"initial_condition":"supplied_continuation_state"})
    return result,{"output":output,"state":next_state}
