# Sub-skill: Unified multimodal artifacts

Part of `mathkernel-mcp`. Use when the same mathematical source should be
visualized and sonified in one self-contained research artifact.

Preferred sequence:

```text
math_projection_create
   -> math_visualize_projection -> viz_id
   -> math_sonify_projection    -> sonification_id
   -> math_research_artifact_create
   -> math_export_research_artifact
```

Both sensory documents inherit the same structured `SourceRef`; automatic
synchronization can therefore link audio intervals and visual blocks without
matching human-readable labels.

The projection records assumptions, evidence references, parameters and
information loss. The artifact layer merges lineage and evidence but never
creates or upgrades mathematical evidence. Visible/audible patterns remain
candidate observations until checked mathematically.
