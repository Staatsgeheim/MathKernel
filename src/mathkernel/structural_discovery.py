# =============================================================================
# MathKernel - Structural symmetry discovery for word expressions
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Constrained exact symmetry discovery over word-expression ASTs.

This is intentionally small and auditable.  It searches a bounded family of
candidate state transformations, substitutes them structurally, canonicalizes
under certified rewrite rules, and reports a symmetry only when both
expressions reduce to the same canonical form.  Numeric probing may rank
candidates, but never upgrades trust to EXACT.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

COMMUTATIVE={"mul","xor","add"}

def Var(name="x"): return ("var",name)
def Const(v): return ("const",int(v))
def Op(name,*args): return (name,*args)

def _key(e): return repr(e)

def substitute(e, variable: str, replacement):
    if e[0]=="var": return replacement if e[1]==variable else e
    if e[0]=="const": return e
    return (e[0],*(substitute(a,variable,replacement) for a in e[1:]))

def canonical(e, word_bits: int):
    op=e[0]; mask=(1<<word_bits)-1
    if op=="var": return e
    if op=="const": return ("const",e[1]&mask)
    args=[canonical(a,word_bits) for a in e[1:]]
    if op=="xor":
        # exact word identities: a xor 0=a, a xor a=0; flatten/sort.
        flat=[]
        for a in args:
            flat.extend(a[1:] if a[0]=="xor" else [a])
        parity={}
        for a in flat: parity[a]=1-parity.get(a,0)
        flat=[a for a,v in parity.items() if v]
        if not flat:return Const(0)
        if len(flat)==1:return flat[0]
        return ("xor",*sorted(flat,key=_key))
    if op in ("mul","add"):
        flat=[]
        for a in args: flat.extend(a[1:] if a[0]==op else [a])
        if op=="mul" and any(a==Const(0) for a in flat): return Const(0)
        ident=Const(1 if op=="mul" else 0)
        flat=[a for a in flat if a!=ident]
        if not flat:return ident
        if len(flat)==1:return flat[0]
        return (op,*sorted(flat,key=_key))
    if op=="fold":
        return ("fold",args[0])
    if op=="rotl":
        return ("rotl",args[0],args[1])
    return (op,*args)

def evaluate(e,x:int,word_bits:int):
    mask=(1<<word_bits)-1; op=e[0]
    if op=="var": return x&mask
    if op=="const": return e[1]&mask
    a=[evaluate(v,x,word_bits) for v in e[1:]]
    if op=="xor":
        r=0
        for v in a:r^=v
        return r&mask
    if op=="add": return sum(a)&mask
    if op=="mul":
        r=1
        for v in a:r=(r*v)&((1<<(2*word_bits))-1)
        return r
    if op=="fold":
        p=a[0]; return ((p&mask)^((p>>word_bits)&mask))&mask
    if op=="low": return a[0]&mask
    if op=="high": return (a[0]>>word_bits)&mask
    if op=="rotl":
        n=a[1]%word_bits; return ((a[0]<<n)|(a[0]>>(word_bits-n)))&mask
    raise ValueError(f"unsupported op {op}")

def candidate_transforms(constants:list[int],word_bits:int,variable="x"):
    x=Var(variable); mask=(1<<word_bits)-1
    out=[("identity",x)]
    for c0 in constants:
        c=c0&mask
        out += [(f"xor({c})",Op("xor",x,Const(c))),
                (f"add({c})",Op("add",x,Const(c))),
                (f"sub({c})",Op("add",x,Const((-c)&mask)))]
    out += [("neg",Op("mul",Const(mask),x))]
    for r in sorted({1,word_bits//4,word_bits//2,3*word_bits//4}):
        if 0<r<word_bits: out.append((f"rotl({r})",Op("rotl",x,Const(r))))
    # de-duplicate structurally
    seen=set(); unique=[]
    for n,t in out:
        c=canonical(t,word_bits)
        if c not in seen: seen.add(c); unique.append((n,c))
    return unique

def discover_symmetries(expr,word_bits:int,constants:list[int],variable="x"):
    """Return exact non-identity symmetries proved by canonical rewrite."""
    base=canonical(expr,word_bits); found=[]
    for name,t in candidate_transforms(constants,word_bits,variable):
        if name=="identity": continue
        transformed=canonical(substitute(expr,variable,t),word_bits)
        if transformed==base:
            found.append({"name":name,"transform":t,"proof":"canonical-rewrite","trust":"exact"})
    return {"symmetries":found,"searched":len(candidate_transforms(constants,word_bits,variable))-1,
            "trust":"exact" if found else "none"}

def folded_mul_x_xor_c(c:int):
    x=Var("x")
    return Op("fold",Op("mul",x,Op("xor",x,Const(c))))


def _transform_name(e):
    op=e[0]
    if op=="var": return e[1]
    if op=="const": return str(e[1])
    return f"{op}(" + ",".join(_transform_name(a) for a in e[1:]) + ")"

def synthesized_transforms(constants:list[int], word_bits:int, variable="x",
                           max_depth:int=2, max_candidates:int=4096):
    """Enumerate a bounded grammar of one/two-operation word transformations."""
    x=Var(variable); mask=(1<<word_bits)-1
    cs=sorted({int(c)&mask for c in constants})
    atoms=[x]
    level1=[]
    for c in cs:
        level1 += [Op("xor",x,Const(c)), Op("add",x,Const(c)),
                   Op("add",x,Const((-c)&mask))]
    level1 += [Op("mul",Const(mask),x)]
    for r in sorted({1,word_bits//4,word_bits//2,3*word_bits//4}):
        if 0<r<word_bits: level1.append(Op("rotl",x,Const(r)))
    levels=[level1]
    if max_depth>=2:
        level2=[]
        for inner in level1:
            for c in cs:
                level2 += [Op("xor",inner,Const(c)),Op("add",inner,Const(c)),
                           Op("add",inner,Const((-c)&mask))]
            for r in sorted({1,word_bits//4,word_bits//2,3*word_bits//4}):
                if 0<r<word_bits: level2.append(Op("rotl",inner,Const(r)))
        levels.append(level2)
    identity=canonical(x,word_bits)
    seen={identity}; out=[]
    # A small deterministic semantic fingerprint removes transformations that
    # are merely unsimplified aliases of identity (e.g. +C then -C, rot16+rot48).
    probes=[0,1,2,3,mask,mask>>1,0x9e3779b97f4a7c15 & mask]
    for level in levels:
        for t in level:
            c=canonical(t,word_bits)
            if c in seen: continue
            if all(evaluate(c,q,word_bits)==q for q in probes):
                seen.add(c); continue
            seen.add(c); out.append(c)
            if len(out)>=max_candidates:return out
    return out

def rank_transform_candidates(expr, word_bits:int, transforms, variable="x",
                              samples:int=1024, seed:int=0x4d4b):
    """Deterministically rank candidates by observed equality rate.

    This is discovery-only evidence.  It never establishes exact trust.
    """
    import random
    rng=random.Random(seed); mask=(1<<word_bits)-1
    xs=[rng.getrandbits(word_bits)&mask for _ in range(samples)]
    base=[evaluate(expr,x,word_bits) for x in xs]
    ranked=[]
    for t in transforms:
        eq=0
        for x,y in zip(xs,base):
            tx=evaluate(t,x,word_bits)
            if evaluate(expr,tx,word_bits)==y: eq+=1
        ranked.append({"transform":t,"name":_transform_name(t),
                       "matches":eq,"samples":samples,"score":eq/samples})
    ranked.sort(key=lambda r:(-r["score"],r["name"]))
    return ranked

def synthesize_symmetries(expr, word_bits:int, constants:list[int], variable="x",
                          max_depth:int=2, probe_samples:int=1024,
                          proof_top_k:int=64):
    """Search a small transformation grammar, rank numerically, prove exactly.

    Exact acceptance requires canonical(substitute(g,S)) == canonical(g).
    """
    transforms=synthesized_transforms(constants,word_bits,variable,max_depth)
    ranked=rank_transform_candidates(expr,word_bits,transforms,variable,probe_samples)
    base=canonical(expr,word_bits); exact=[]
    for row in ranked[:proof_top_k]:
        transformed=canonical(substitute(expr,variable,row["transform"]),word_bits)
        if transformed==base:
            exact.append({**row,"proof":"canonical-rewrite","trust":"exact"})
    return {"candidate_count":len(transforms),"ranked":ranked[:proof_top_k],
            "exact_symmetries":exact,
            "probe_trust":"numeric","proof_trust":"exact" if exact else "none"}


def classify_simple_transform(t, variable="x"):
    """Recognize exact affine-access-compatible transforms after canonicalization."""
    if t[0] in ("xor","add") and len(t)==3:
        consts=[a[1] for a in t[1:] if a[0]=="const"]
        vars_=[a for a in t[1:] if a==Var(variable)]
        if len(consts)==1 and len(vars_)==1:
            return {"kind":t[0],"constant":consts[0]}
    return None


def evaluate_vector_expr(e, state: tuple[int,...], word_bits:int):
    """Evaluate a word expression whose variables are x0,x1,... over a state tuple."""
    mask=(1<<word_bits)-1
    op=e[0]
    if op=="var":
        name=e[1]
        if name.startswith("x") and name[1:].isdigit():
            return state[int(name[1:])] & mask
        raise ValueError(f"unknown vector variable {name}")
    if op=="const": return e[1]&mask
    a=[evaluate_vector_expr(v,state,word_bits) for v in e[1:]]
    if op=="xor":
        r=0
        for v in a:r^=v
        return r&mask
    if op=="add": return sum(a)&mask
    if op=="mul":
        r=1
        for v in a:r*=v
        return r
    if op=="fold":
        p=a[0]; return ((p&mask)^((p>>word_bits)&mask))&mask
    if op=="low": return a[0]&mask
    if op=="high": return (a[0]>>word_bits)&mask
    if op=="rotl":
        n=a[1]%word_bits
        return ((a[0]<<n)|(a[0]>>(word_bits-n)))&mask if n else a[0]&mask
    if op=="shl": return (a[0]<<a[1])&mask
    if op=="shr": return (a[0]>>a[1])&mask
    if op=="asr":
        v=a[0]
        sv=v-(1<<word_bits) if v&(1<<(word_bits-1)) else v
        return (sv>>a[1])&mask
    raise ValueError(f"unsupported vector op {op}")

def vector_candidate_transforms(constants:list[int],word_bits:int,words:int=2,
                                max_candidates:int=4096):
    """Bounded grammar for multiword state transformations."""
    if words!=2: raise ValueError("current vector synthesizer supports two words")
    mask=(1<<word_bits)-1; x0=Var("x0");x1=Var("x1")
    out=[("swap",(x1,x0))]
    cs=sorted({int(c)&mask for c in constants})
    for c in cs:
        out += [
          (f"xor0({c})",(Op("xor",x0,Const(c)),x1)),
          (f"xor1({c})",(x0,Op("xor",x1,Const(c)))),
          (f"xor_both({c})",(Op("xor",x0,Const(c)),Op("xor",x1,Const(c)))),
          (f"add0({c})",(Op("add",x0,Const(c)),x1)),
          (f"add1({c})",(x0,Op("add",x1,Const(c)))),
        ]
    out += [
      ("cross01",(Op("xor",x0,x1),x1)),
      ("cross10",(x0,Op("xor",x1,x0))),
      ("cross_both",(Op("xor",x0,x1),Op("xor",x1,x0))),
    ]
    for r in sorted({1,word_bits//4,word_bits//2}):
        if 0<r<word_bits:
            out += [(f"rot0({r})",(Op("rotl",x0,Const(r)),x1)),
                    (f"rot1({r})",(x0,Op("rotl",x1,Const(r))))]
    # compose base transformations pairwise, bounded
    base=list(out)
    def subst_pair(expr,pair):
        e=substitute(expr,"x0",pair[0])
        return substitute(e,"x1",pair[1])
    for n1,t1 in base:
        for n2,t2 in base:
            out.append((f"{n2}∘{n1}",(subst_pair(t2[0],t1),subst_pair(t2[1],t1))))
            if len(out)>=max_candidates:break
        if len(out)>=max_candidates:break
    # semantic de-duplication using deterministic probes
    probes=[(0,0),(1,0),(0,1),(1,1),(mask,0),(0,mask),(0x55&mask,0xa3&mask)]
    seen=set();unique=[]
    for name,t in out:
        fp=tuple((evaluate_vector_expr(t[0],q,word_bits),
                  evaluate_vector_expr(t[1],q,word_bits)) for q in probes)
        ident=tuple(probes)
        if fp==ident or fp in seen:continue
        seen.add(fp);unique.append((name,t))
    return unique

def apply_vector_transform(t,state,word_bits):
    return tuple(evaluate_vector_expr(e,state,word_bits) for e in t)

def synthesize_vector_symmetries(outputs:list, word_bits:int, constants:list[int],
                                 probe_samples:int=2048, proof_exhaustive_bits:int=16,
                                 top_k:int=64, seed:int=0x564543):
    """Rank two-word transformations and prove reduced-state symmetries exactly."""
    import random
    rng=random.Random(seed); mask=(1<<word_bits)-1
    candidates=vector_candidate_transforms(constants,word_bits)
    states=[(rng.getrandbits(word_bits),rng.getrandbits(word_bits))
            for _ in range(probe_samples)]
    def obs(st):
        vals=tuple(evaluate_vector_expr(e,st,word_bits) for e in outputs)
        return vals[0] if len(vals)==1 else vals
    base=[obs(q) for q in states];ranked=[]
    for name,t in candidates:
        eq=sum(obs(apply_vector_transform(t,q,word_bits))==b for q,b in zip(states,base))
        ranked.append({"name":name,"transform":t,"score":eq/probe_samples,
                       "matches":eq,"samples":probe_samples})
    ranked.sort(key=lambda z:(-z["score"],z["name"]))
    exact=[]
    # Full exhaustive proof is feasible when total state bits <= threshold.
    if 2*word_bits<=proof_exhaustive_bits:
        allstates=((a,b) for a in range(1<<word_bits) for b in range(1<<word_bits))
        for row in ranked[:top_k]:
            if all(obs(apply_vector_transform(row["transform"],q,word_bits))==obs(q)
                   for q in allstates):
                exact.append({**row,"trust":"exact","proof":"exhaustive-finite-state"})
            allstates=((a,b) for a in range(1<<word_bits) for b in range(1<<word_bits))
    return {"candidate_count":len(candidates),"ranked":ranked[:top_k],
            "exact_symmetries":exact,"ranking_trust":"numeric",
            "proof_trust":"exact" if exact else "none"}
