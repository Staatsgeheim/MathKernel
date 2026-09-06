# =============================================================================
# MathKernel Sonify - offline WebAudio sonification inspector
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""Single-file offline WebAudio inspector for SonificationDocument."""
from __future__ import annotations
import hashlib, html, json
from pathlib import Path
from .webaudio import webaudio_payload
from ..models import SonificationDocument

def render_html(doc: SonificationDocument) -> str:
    payload=json.dumps(webaudio_payload(doc),sort_keys=True,separators=(',',':'),ensure_ascii=True).replace('</','<\\/')
    title=html.escape(doc.title)
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src blob:; connect-src 'none'">
<title>{title}</title><style>body{{font:14px system-ui;margin:2rem;max-width:1000px}}button{{margin-right:.5rem}}pre{{white-space:pre-wrap;background:#eee;padding:1rem;max-height:24rem;overflow:auto}}.warn{{font-weight:600}}</style></head><body>
<h1>{title}</h1><p class="warn">Scientific sonification: audible patterns are candidate observations, not proof.</p>
<p><button id="play">Play</button><button id="stop">Stop</button> Duration: <span id="dur"></span>s</p><details><summary>Mapping / provenance</summary><pre id="meta"></pre></details>
<script id="mathkernel-sonification" type="application/json">{payload}</script><script>
const D=JSON.parse(document.getElementById('mathkernel-sonification').textContent);let ctx=null,nodes=[];
document.getElementById('dur').textContent=D.duration.toFixed(3);document.getElementById('meta').textContent=JSON.stringify(D,null,2);
function map(v,s){{let t=s.type||'identity';if(t==='identity')return v;if(t==='linear')return (s.offset||0)+(s.scale??1)*v;if(t==='harmonic')return (s.fundamental||110)*v;if(t==='abs')return Math.abs(v);if(t==='clamp')return Math.min(s.max,Math.max(s.min,v));if(t==='normalize'){{let u=s.source_max===s.source_min ? .5 :(v-s.source_min)/(s.source_max-s.source_min);return s.target_min+u*(s.target_max-s.target_min)}}if(t==='log_map'){{let u=s.source_max===s.source_min ? .5 :Math.max(0,Math.min(1,(v-s.source_min)/(s.source_max-s.source_min)));return s.target_min*Math.pow(s.target_max/s.target_min,u)}}throw Error('unsupported transform '+t)}}
function stop(){{for(const n of nodes)try{{n.stop()}}catch(e){{}}nodes=[];if(ctx){{ctx.close();ctx=null}}}}
async function play(){{stop();ctx=new AudioContext({{sampleRate:D.render.sample_rate}});const base=ctx.currentTime+.03;for(const tr of D.tracks){{for(const e of tr.events){{let f=e.values.frequency??440,g=e.values.gain??1,p=e.values.coefficient_phase??e.values.phase??0;for(const id of tr.mapping_refs){{const m=D.mappings[id];if(!m||!(m.source in e.values))continue;const v=map(e.values[m.source],m.transform);if(m.target==='frequency')f=v;else if(m.target==='gain')g=v;else if(m.target==='phase')p=v}}if(f<=0||f>=ctx.sampleRate/2)continue;const o=ctx.createOscillator(),gn=ctx.createGain(),pn=ctx.createStereoPanner();o.frequency.value=f;gn.gain.value=Math.min(D.render.peak_limit,Math.abs(g*(tr.gain??1))*.25);pn.pan.value=tr.pan??0;o.connect(gn).connect(pn).connect(ctx.destination);o.start(base+e.time);o.stop(base+e.time+e.duration);nodes.push(o)}}}}
document.getElementById('play').onclick=play;document.getElementById('stop').onclick=stop;
</script></body></html>'''

def export_html(doc: SonificationDocument,path: str|Path)->dict:
    text=render_html(doc); p=Path(path); p.write_text(text,encoding='utf-8',newline='\n'); raw=text.encode()
    return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'duration':doc.duration(),'portable':True}
