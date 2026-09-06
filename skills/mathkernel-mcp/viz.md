# Sub-skill: Visualization and multimodal projections

Part of `mathkernel-mcp`. Use when plotting mathematical objects/results or
exporting evidence-carrying visual artifacts.

## Preferred tool workflow

1. `math_projection_catalog()` — discover canonical projection families.
2. `math_projection_create(kind, payload, ...)` — create a typed presentation
   projection with source lineage, assumptions, evidence refs, projection
   parameters and declared information loss. Shortcut: pass only
   `source_object_id` (an object from `math_object_create`/`math_apply`, or an
   `rv_id` from `math_prob_rv_create`) and a registered domain adapter chooses
   the kind and payload automatically; its presentation choices (sampling
   grids, magnitude-only spectra, dropped channels) are declared in the
   projection's `parameters`/`information_loss`.
3. `math_visualize_projection(projection_id, title?)` — generate a stored
   `VisualizationDocument` and return `viz_id`.
4. `math_export_artifact(viz_id, path, ...)` — write portable HTML.

Even shorter for visuals only: `math_visualize(object_id=...)` runs the same
adapter registry and returns a `viz_id` directly. Unregistered object types
fail with a typed error — they never get an invented view.

Higher-dimensional inputs (>3D) require explicit `parameters` containing at
least `input_dimension`, `output_dimension` and `method`, plus
`information_loss=["projection"]`. Do not let the frontend invent a basis.

Canonical families cover scalar/vector fields, point sets/clouds, curves,
surfaces, distributions, matrices/tensors, graphs/evidence DAGs, spectra,
regions/sets, meshes/complexes, complex fields, optimization, statistical
inference, finite dynamics, ODE/PDE solutions, finite-field structures,
relation geometry, partitions/piecewise objects, expression/certificate trees,
quantities, ensembles and higher-dimensional projections.

`math_visualize(...)` remains available for direct low-level composition and
legacy matrix/step/system/inline sources. Use projection tools for domain
objects so audio and visualization can share lineage.

Visible structure is not new evidence. Preserve certified enclosures as
regions and record all lossy projection choices.
