# =============================================================================
# MathKernel Sonify - evidence-aware scientific sonification
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
"""mathkernel-sonify: evidence-aware scientific sonification for MathKernel."""
from .models import SONIFY_SCHEMA, Mapping, SonificationEvent, AudioTrack, RenderConfig, SonificationDocument
from .mapping import apply_transform
from .adapters import harmonic_sonification, scan_sonification, compare_sonification, projection_sonification
from .renderers import render_float, pcm_bytes, write_wav, webaudio_payload, render_html, export_html
from .artifact import attach_to_artifact, synchronize

def sonify(data, *, mode="auto", **kwargs):
    if mode in ("harmonic","fourier"): return harmonic_sonification(data,**kwargs)
    if mode in ("scan","sequence","auto"): return scan_sonification(data,**kwargs)
    raise ValueError(f"unsupported sonification mode {mode!r}")

def sonify_compare(prediction, observation, *, mode="stereo", **kwargs):
    return compare_sonification(prediction,observation,mode=mode,**kwargs)

def sonify_residual(prediction, observation, **kwargs):
    return compare_sonification(prediction,observation,mode="residual",**kwargs)

__all__=["SONIFY_SCHEMA","Mapping","SonificationEvent","AudioTrack","RenderConfig","SonificationDocument","apply_transform","harmonic_sonification","scan_sonification","compare_sonification","projection_sonification","sonify","sonify_compare","sonify_residual","render_float","pcm_bytes","write_wav","webaudio_payload","render_html","export_html","attach_to_artifact","synchronize"]
