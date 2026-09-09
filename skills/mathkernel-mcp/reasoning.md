# Sub-skill: Reasoning, proof, and verified codegen

Part of the `mathkernel-mcp` skill. Load when the task is proving
equivalence, finding counterexamples, running multi-step derivations, or
generating verified code.

## One-shot reasoning

`math_reason(expr_id, context_id, formal=true, max_steps=32)` — runs the
obligation-DAG planner end to end: decompose into obligations, discharge
each with the appropriate engine (SymPy simplify, Z3 SMT, Lean when
`formal=true`), and return a verified result with the full derivation.
Use this as the default for "prove/show/verify this identity or relation".

## Manual plan control

1. `math_plan(expr_id, context_id, solve_for)` → `plan_id` + obligation DAG.
2. `math_plan_get(plan_id)` inspects the decomposition.
3. `math_execute_plan(plan_id, formal, max_steps)` → `execution_id`.
4. `math_execution_get(execution_id)` polls the result.

Use the manual flow when you need to inspect the decomposition before
committing to execution, or when a previous execution failed and you want
to adjust `max_steps`/`formal`.

## Equivalence and counterexamples

- `math_prove_equivalence(left, right, context_id, formal)` — two
  expressions (raw strings, parsed internally); returns proved/disproved/
  unknown with engine evidence.
- `math_counterexample(left, right, context_id)` — searches for a
  falsifying assignment instead of proving.

Approximate inputs: if either side or a required assumption contains a decimal literal, the result
is capped at `numeric` trust, no Lean certificate is issued (formal backends
encode decimals as exact rationals — a different statement), and
`math_counterexample` returns `unknown` with a warning. Rewrite decimals as
exact rationals (`1/10`, not `0.1`) for proof-grade evidence.

Declined independent verifiers are diagnostics, not required proof dependencies.
Only checked scripts appear in `lean_certificate`; `lean_candidate` carries
`checked: false` and proves nothing. Already-refuted statements skip Lean and
contain neither artifact. Lean installation requires explicit operator setup
with `mathkernel-lean-setup`; ordinary calls never download a toolchain.
Solution reasoning likewise keeps only checked scripts in `candidate_certificates`;
unchecked work appears in `candidate_attempts` with `checked: false`.

## Verified code generation

1. `math_codegen(expr_id, language, mode, target, variable)` — language:
   `typescript` | `python` | `rust`; target: `evaluate` | `solve` |
   `constraint`. Emits capability-interfaced generic code (field
   operations only — no factorial/gamma/binomial/pi).
2. `math_verify_code(artifact_id, checks)` — compiler/type checking plus a
   symbolic roundtrip (Python) of the artifact against the source
   expression. Overall trust is the weakest evidence: unavailable checks
   (e.g. no Rust toolchain) cap the result below `exact`.
3. `math_execute_code(artifact_id, inputs)` — sandboxed numeric run
   (isolated subprocess, timeout; numeric evidence only).

Never skip step 2: generated code is unverified until `math_verify_code`
passes.

## Audit trail

Every result embeds derivation steps with `step_id`s.
`math_derivation_get(step_id)` fetches one step;
`math_derivation_trace(step_id)` reconstructs the full ancestry — use it
when the user asks "why/how did you conclude this".
