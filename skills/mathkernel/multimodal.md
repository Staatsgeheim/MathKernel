# Sub-skill: Unified multimodal research artifacts

Part of the `mathkernel` skill. Load when visualization and audio should
represent the same mathematical source in one evidence-carrying artifact.

## Preferred architecture

```text
MathResult / typed object
       -> MultimodalProjection
          |             |
          v             v
 VisualizationDocument  SonificationDocument
          \             /
           MathKernelArtifact
             -> portable HTML
```

`mathkernel_projection` is the shared semantic boundary. It records source
lineage, assumptions, evidence references, projection parameters and declared
information loss. `mathkernel_viz` and `mathkernel_sonify` are sibling sensory
frontends over that object.

Because both documents carry the same structured `SourceRef`,
`mathkernel_multimodal.build_artifact(..., auto_synchronize=True)` can derive
visual/audio synchronization links without guessing from labels.

## Kernel workflow

```python
p = kernel.projection_create("relation_geometry", {
    "eigenvalues": eigenvalues,
    "matrix": information_matrix,
}, trust="numeric")
pid = p.data["projection_id"]

v = kernel.viz_projection(pid)
a = kernel.sonify_projection(pid)

artifact = kernel.research_artifact_create(
    title="Relation geometry",
    viz_ids=[v.data["viz_id"]],
    sonification_ids=[a.data["sonification_id"]],
)
kernel.research_artifact_export(artifact.data["artifact_id"], "relation.html")
```

## Rules

- The multimodal layer packages evidence; it does not create it.
- Trust is never upgraded by presentation.
- Projection loss is separate from mathematical-source trust.
- High-dimensional projection matrices/bases/methods must be explicit.
- Perceptual annotations are candidate observations until quantitatively
  checked against the mathematical source.
- Prefer one source projection feeding both modalities over independently
  constructing unrelated visual/audio mappings.
