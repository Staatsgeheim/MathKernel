# Units, persistence, replay, fuzzing (MCP)

## Units (exact trust)

```
math_unit_check(expr_id, units={"v": "m/s", "t": "s"})  # SI dimension
math_unit_convert("36", "km/h", "m/s")                  # exact rational
math_unit_simplify("kg*m/s^2")                          # dimension + scale
```

Dimensional inconsistencies are hard errors. Unit syntax: `kg*m/s^2`, `km/h`.

## Persistence and replay

```
math_store_status()        # is MATHKERNEL_STORE_PATH active?
math_replay(step_id)       # topological provenance chain, integrity-checked
```

`math_yolo_settings` can set `MATHKERNEL_STORE_PATH` (and every other
`MATHKERNEL_*` knob) only when `MATHKERNEL_YOLO_MODE=true`. Null/empty
clears the store path and closes the live SQLite handle.

## Certified enclosure and fuzzing

```
math_certified_enclose(expr_id, "x", "0", "1")   # Arb or mpmath.iv
math_fuzz_differential(n=100, variables=["x"], seed=0)
```

`math_fuzz_differential` compares the high-precision and float64 tiers on
random expressions; `status: conflict` means a real tier disagreement — do
not ignore it.
