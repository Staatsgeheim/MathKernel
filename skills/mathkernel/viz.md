# Sub-skill: Visualization and multimodal projections

Part of the `mathkernel` skill. Load when creating plots, evidence diagrams,
interactive dashboards, or portable visualization artifacts.

## Core rule

Visualization is a presentation of an existing mathematical object; it does
not create mathematical evidence. Prefer the shared
`mathkernel_projection.MultimodalProjection` layer for domain objects so the
same source lineage, assumptions, explicit projection parameters and declared
information loss can also be consumed by sonification.

```text
mathematical object/result
        -> MultimodalProjection
             -> mathkernel_viz VisualizationDocument
             -> SVG / Matplotlib / portable HTML (+ Three.js when needed)
```

## Canonical projection families

`mathkernel_projection.projection_catalog()` lists the live catalog. It covers:

- scalar/vector fields, point sets/clouds, curves, surfaces and sequences;
- distributions, matrices, tensors, spectra and complex fields;
- graphs, finite dynamics, evidence DAGs, expression trees and certificate trees;
- regions, implicit sets, sets, partitions and piecewise objects;
- meshes, geometric complexes, ODE/PDE solutions and geometry objects;
- optimization traces/feasible geometry and statistical-inference objects;
- finite-field/GF(2) structures and relation/information geometry;
- quantities/units, ensembles, and explicit higher-dimensional projections.

Higher-dimensional data must name the projection method and output dimension;
for input dimension >3 the projection must declare
`information_loss=["projection"]`. Never let a renderer silently choose PCA,
a coordinate subset, tensor slice, aggregation or basis.

## Python API

```python
import mathkernel_projection as mkp
import mathkernel_viz as viz

p = mkp.create_projection(
    "matrix",
    {"matrix": [[1, 2], [3, 4]]},
    trust="exact",
)
doc = viz.from_projection(p)
viz.export_html(doc, "matrix.html")
```

Kernel facade:

```python
r = kernel.projection_create("mesh", {
    "vertices": [[0, 0], [1, 0], [0, 1]],
    "cells": [[0, 1, 2]],
}, trust="exact")
pid = r.data["projection_id"]
visual = kernel.viz_projection(pid)
```

Automatic adaptation: the registry in `mathkernel_projection.result_adapters`
maps stored typed objects onto canonical families. Pass only
`source_object_id` to `kernel.projection_create`, or `object_id` to
`kernel.viz_create`, and the registered adapter derives kind and payload —
covering signals/spectra/filters, pole-zero maps, frequency responses, root
loci, time responses, distributions, statistical samples and fits,
survival/time-series objects, graphs and traversal trees, optimization
results, ODE/SDE ensembles, FEM meshes/solutions/indicators, PDE grids,
geometry objects, generating functions, Cayley tables, contours, singularity
maps, group partitions, counts and unit quantities. Adapters never recompute
mathematics or upgrade trust; sampling windows and dropped channels are
declared in `parameters`/`information_loss`. Unregistered object types raise
a typed error. Register custom mappings with
`register_model_adapter`/`register_engine_adapter`/`register_shape_adapter`.

The visualization document records projection id/kind, parameters,
information loss, evidence references and assumptions. Generated datasets and
series share the projection's structured `SourceRef`, enabling cross-modal
synchronization in unified artifacts.

## Generic visualization building blocks

`mathkernel_viz.blocks` remains available for low-level composition:

- `plot2d`, `histogram`, `heatmap`, `dag`;
- `point_cloud_3d`, `trajectory_3d`, `surface_3d`, `vector_field_3d`;
- `metric_grid`, `data_table`, `text`, `select`.

Use these when composing a custom dashboard. For domain semantics, prefer a
canonical projection and `from_projection` instead of inventing a new renderer.

## Evidence and interpretation

- Do not promote trust because an image looks convincing.
- Certified intervals/regions remain enclosures, not point estimates.
- A complex field's magnitude/phase or real/imaginary panels are declared
  representations of the same source.
- Evidence graphs should show claim/evidence/assumption ancestry where useful;
  they visualize the evidence model but are not themselves proof.
- For high-dimensional data, store the actual projection/basis/matrix in
  `parameters` whenever available.
- A visualization may be lossy even when the underlying result is exact; the
  mathematical source trust and presentation information loss are separate.
