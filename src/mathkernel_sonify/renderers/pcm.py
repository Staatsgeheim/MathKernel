# =============================================================================
# MathKernel Sonify - deterministic PCM/WAV renderer
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Deterministic additive-synthesis PCM renderer."""
from __future__ import annotations
import math, struct, wave, hashlib
from pathlib import Path
from ..models import SonificationDocument
from ..mapping import apply_transform

def render_float(doc: SonificationDocument) -> list[tuple[float,float]]:
    sr=doc.render.sample_rate; n=max(0, int(math.ceil(doc.duration()*sr)))
    out=[[0.0,0.0] for _ in range(n)]
    nyquist=sr/2
    for track in doc.tracks:
        pan=max(-1.0,min(1.0,track.pan)); lg=math.sqrt((1-pan)/2); rg=math.sqrt((1+pan)/2)
        maps=[doc.mappings[m] for m in track.mapping_refs if m in doc.mappings]
        for ev in track.events:
            vals=dict(ev.values); freq=vals.get("frequency",440.0); gain=vals.get("gain",1.0); phase=vals.get("phase",0.0)
            for m in maps:
                if m.source in vals:
                    v=apply_transform(vals[m.source],m.transform)
                    if m.target=="frequency": freq=v
                    elif m.target=="gain": gain=v
                    elif m.target=="phase": phase=v
            if not (0 <= freq < nyquist):
                raise ValueError(f"frequency {freq} Hz violates Nyquist limit {nyquist} Hz; use an explicit audible-domain mapping")
            start=max(0,int(round(ev.time*sr))); end=min(n,int(round((ev.time+ev.duration)*sr)))
            g=float(gain)*track.gain
            for i in range(start,end):
                s=g*math.sin(2*math.pi*freq*((i-start)/sr)+phase)
                out[i][0]+=s*lg; out[i][1]+=s*rg
    peak=max((max(abs(a),abs(b)) for a,b in out),default=0.0)
    scale=1.0
    if doc.render.normalization=="global" and peak>0: scale=min(1.0,doc.render.peak_limit/peak)
    elif peak>doc.render.peak_limit: scale=doc.render.peak_limit/peak
    return [(a*scale,b*scale) for a,b in out]

def pcm_bytes(doc: SonificationDocument) -> bytes:
    frames=render_float(doc); bits=doc.render.bit_depth; channels=doc.render.channels
    buf=bytearray()
    for l,r in frames:
        vals=(l,) if channels==1 else (l,r)
        for x in vals:
            x=max(-1.0,min(1.0,x))
            if bits==16: buf.extend(struct.pack('<h',round(x*32767)))
            elif bits==24:
                v=round(x*8388607); v=v if v>=0 else (1<<24)+v; buf.extend(bytes((v&255,(v>>8)&255,(v>>16)&255)))
            else: buf.extend(struct.pack('<i',round(x*2147483647)))
    return bytes(buf)

def write_wav(doc: SonificationDocument, path: str|Path) -> dict:
    raw=pcm_bytes(doc); p=Path(path); sw=doc.render.bit_depth//8
    with wave.open(str(p),'wb') as w:
        w.setnchannels(doc.render.channels); w.setsampwidth(sw); w.setframerate(doc.render.sample_rate); w.writeframes(raw)
    data=p.read_bytes()
    return {"path":str(p),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"duration":doc.duration(),"sample_rate":doc.render.sample_rate}
