# Sub-skill: Scientific sonification and multimodal projections

Part of `mathkernel-mcp`. Use when mathematical structure should be explored
audibly or exported as deterministic WAV/WebAudio.

## Preferred tool workflow

1. Build or reuse a canonical projection with `math_projection_create`.
2. `math_sonify_projection(projection_id, mode="auto", options=...)`.
3. Inspect with `math_sonification_describe` before interpreting the sound.
4. Export with `math_export_audio` or combine with a visualization in a
   research artifact.

The projection sonifier covers every canonical projection family. Structured
objects are never silently flattened: the returned document records its
`acoustic_extraction`, such as row-major matrix scan, tensor slice scan,
vector/state norm, graph degree, mesh edge length, boundary segment length,
optimization trace, inference distribution, relation eigenvalues, or ensemble
member ordering.

Legacy `math_sonify(data, mode=scan|harmonic|fourier)` and
`math_sonify_compare(...)` remain available for already-scalar sequences and
spectra.

Audible patterns are candidate observations only. Deterministic rendering is
reproducibility of the encoding, not proof of the source claim. Never hide
normalization, projection, aggregation, musical quantization or other
perceptual transforms.
