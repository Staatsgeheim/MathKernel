# Sub-skill: Scientific sonification and multimodal projections

Part of the `mathkernel` skill. Load when converting mathematical structure to
audio, comparing prediction/observation acoustically, or exporting WAV/audio
inside a unified research artifact.

## Core rule

Audio is an explicit projection of mathematical structure, never proof.
Prefer `MultimodalProjection` as the source so audio and visualization share
lineage and projection semantics.

```text
mathematical object/result
        -> MultimodalProjection
             -> explicit acoustic extraction
             -> SonificationDocument
             -> deterministic WAV / WebAudio
```

Structured objects are never silently flattened. Every reduction or ordering
is named in transformation provenance and document metadata, for example:

- matrix -> row-major scan;
- tensor -> declared slice/storage-order scan;
- vector field -> Euclidean vector-magnitude scan;
- graph/evidence tree -> node-degree scan;
- mesh/geometric complex -> edge-length scan;
- region -> boundary-segment-length scan;
- complex field -> magnitude to gain, phase to oscillator phase;
- ODE/dynamical vector state -> Euclidean state norm over time;
- optimization -> objective trace;
- statistical inference -> null/bootstrap/sample distribution;
- relation geometry -> information/relation eigenvalue scan;
- ensemble -> declared member order then sample order.

These are presentation choices. If another mapping is scientifically more
appropriate, create a different projection or adapter and record it.

## Python API

```python
import mathkernel_projection as mkp
import mathkernel_sonify as son

p = mkp.create_projection(
    "spectrum",
    {"values": amplitudes, "phase": phases},
    trust="numeric",
)
doc = son.projection_sonification(p)
son.write_wav(doc, "spectrum.wav")
```

Kernel facade:

```python
r = kernel.projection_create("matrix", {"matrix": matrix}, trust="numeric")
pid = r.data["projection_id"]
audio = kernel.sonify_projection(pid, options={"seconds_per_item": 0.03})
```

Legacy direct sequence/spectrum APIs remain valid:

```python
son.sonify(values, mode="scan")
son.sonify(amplitudes, mode="harmonic", phases=phases)
son.sonify_compare(prediction, observation, mode="stereo")
```

## Interpretation discipline

- Audible patterns are candidate observations only.
- Keep prediction and observation roles separate and on a common scale.
- Deterministic rendering makes the audio reproducible; it does not make
  numeric mathematics exact.
- Reject out-of-Nyquist mappings rather than silently aliasing.
- Do not hide normalization, resampling, clipping, musical quantization,
  reverb, compression or other perceptual transforms.
- If information is dropped (norm, slice, projection, aggregation), ensure the
  transformation metadata says so.
