# Units and dimensional analysis

Dimensions are exact 7-vector SI exponents (L, M, T, I, Θ, N, J) with
`Fraction` components. Conversions are exact rational arithmetic. Dimensional
inconsistencies are hard errors, never warnings. All results are `exact`.

```python
kernel.unit_check(expr_id, {"v": "m/s", "t": "s"})  # -> dimension "L"
kernel.unit_convert("36", "km/h", "m/s")            # -> "10" (exact)
kernel.unit_simplify("kg*m/s^2")                    # -> dimension of N
```

- `unit_check` walks MathIR: add/sub require equal dimensions, mul/div
  combine them, powers scale them (integer/rational exponents only for
  dimensioned bases), transcendental functions require dimensionless
  arguments. The result is recorded in `SemanticMeta.units`.
- Unit expressions: `name[*name[^int]...] [/ ...]` — e.g. `"kg*m/s^2"`,
  `"km/h"`, `"mL"`. No parentheses.
- Registry: SI base (m, kg, s, A, K, mol, cd), derived (N, Pa, J, W, Hz, C,
  V, ohm), prefixed (km, cm, mm, g, mL), time (min, h, day), others (L, bar,
  cal, eV, mph). Scales are exact by SI definition.

There is deliberately no numeric tier here — dimensional analysis is
metadata-level exact work.
