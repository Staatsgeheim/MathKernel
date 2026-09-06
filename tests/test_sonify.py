# =============================================================================
# MathKernel - sonification tests
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
import hashlib, math, wave
import pytest
import mathkernel_sonify as son
from mathkernel_artifacts import MathKernelArtifact

def test_harmonic_mapping_is_explicit():
    d=son.harmonic_sonification([1,.5],[0,.25],fundamental=100,duration=.02,sample_rate=8000)
    assert d.mappings['harmonic-frequency'].transform == {'type':'harmonic','fundamental':100}
    assert d.transformations[0].operation == 'harmonic_additive_synthesis'
    assert d.tracks[0].events[1].values['harmonic'] == 2

def test_pcm_is_deterministic():
    d=son.harmonic_sonification([1,.2],fundamental=100,duration=.03,sample_rate=8000)
    a=son.pcm_bytes(d); b=son.pcm_bytes(d)
    assert a == b and hashlib.sha256(a).digest()==hashlib.sha256(b).digest()

def test_nyquist_refuses_aliasing():
    d=son.harmonic_sonification([1,1,1],fundamental=2000,duration=.01,sample_rate=8000)
    with pytest.raises(ValueError,match='Nyquist'): son.pcm_bytes(d)

def test_scan_has_source_time_correspondence():
    d=son.scan_sonification([1,2,4],seconds_per_item=.25,sample_rate=8000)
    assert [(e.time,e.duration,e.label) for e in d.tracks[0].events] == [(0,.25,'0'),(.25,.25,'1'),(.5,.25,'2')]

def test_comparison_common_scale_and_roles():
    d=son.sonify_compare([0,1],[0,100],mode='stereo',sample_rate=8000)
    assert [t.role for t in d.tracks] == ['prediction','observation']
    assert [t.pan for t in d.tracks] == [-1,1]
    assert d.mappings['value-frequency'].transform['source_max']==100

def test_residual_is_difference():
    d=son.sonify_residual([1,2,3],[2,4,8],sample_rate=8000)
    assert [e.values['value'] for e in d.tracks[0].events] == [1,2,5]

def test_wav_export(tmp_path):
    d=son.harmonic_sonification([1],fundamental=200,duration=.02,sample_rate=8000)
    info=son.write_wav(d,tmp_path/'a.wav')
    with wave.open(info['path'],'rb') as w:
        assert w.getframerate()==8000 and w.getnchannels()==2 and w.getsampwidth()==3
    assert len(info['sha256'])==64

def test_artifact_bridge_preserves_semantics():
    d=son.scan_sonification([1,2,3],sample_rate=8000)
    a=son.attach_to_artifact(MathKernelArtifact(artifact_id='x'),d)
    assert len(a.sonifications)==1 and a.transformations[-1].purpose=='encoding'
    assert a.sources

def test_webaudio_payload_contains_mapping_not_code():
    d=son.scan_sonification([1,2],sample_rate=8000)
    p=son.webaudio_payload(d)
    assert p['mappings']['value-frequency']['transform']['type']=='log_map'
    assert 'javascript' not in str(p).lower()

def test_portable_html_is_self_contained_and_escaped(tmp_path):
    d=son.scan_sonification([1,2,3],title='x </script><script>alert(1)</script>',sample_rate=8000)
    info=son.export_html(d,tmp_path/'s.html'); text=(tmp_path/'s.html').read_text()
    assert info['portable'] and 'AudioContext' in text and 'https://' not in text
    assert '<\\/script>' in text
