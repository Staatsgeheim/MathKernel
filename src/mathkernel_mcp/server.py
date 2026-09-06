# =============================================================================
# MathKernel - server
# Copyright (c) 2026 Maarten Boone
# SPDX-License-Identifier: MIT
# =============================================================================
from __future__ import annotations

import json

from mathkernel.kernel import MathKernel

kernel = MathKernel()

try:
    from fastmcp import FastMCP
except ImportError as exc:
    FastMCP = None
    _import_error = exc

if FastMCP is not None:
    mcp = FastMCP(
        name="MathKernel MCP",
        instructions="""\
MathKernel MCP is a trustworthy multi-engine mathematics server. Core workflow:

1. DISCOVER: call math_capabilities for a compact domain/engine summary,
   then page math_capability_query (limit/offset) for the relevant operations.
   Request detail="full" only when necessary. Oversized responses retain
   their complete evidence in math_result_resource_get; concatenate content
   pages and verify their sha256 before interpreting the full result. Use math_capability_query to filter by
   domain, input/output type, operation, trust level, verification method, or
   engine.
2. PARSE: math_parse (ASCII) or math_parse_latex (LaTeX) returns an expr_id;
   chain expr_ids into math_solve, math_simplify, math_differentiate,
   math_integrate, math_limit, math_series, math_summation, math_product and
   the math_matrix_* tools. Use math_context_create to attach symbol
   domains/assumptions (e.g. x positive, n integer) and pass context_id.
3. TRUST: every result carries evidence_bundle, claim_evidence,
   semantic_status, and a conservative trust summary. formal = mechanically checked
   proof certificate; exact = exact computation or independently established
   mathematical fact; symbolic = symbolic-engine result (strong evidence, not
   independent proof); interval_certified = rigorous enclosure (use
   math_interval_evaluate); numeric = numerical evidence. Never present
   symbolic or numeric output as proof. Decimal literals are approximate:
   any expression containing one is capped at numeric trust, and formal
   certificates / exact counterexamples are refused for approximate inputs —
   rewrite decimals as exact rationals (1/10, not 0.1) when proof-grade
   evidence is needed. Inspect evidence for the requested claim: a producer's
   justified_trust cannot override weaker ancestry, and an unverified proof or
   certificate supports only unknown. Preserve does_not_exist, undefined,
   infeasible, unsupported, and unknown as distinct outcomes.
4. REASON: math_reason / math_plan + math_execute_plan run the obligation-DAG
   planner with Z3/Lean verification; math_prove_equivalence and
   math_counterexample handle equivalence queries directly.
5. LONG SWEEPS: math_collatz_sieve and math_cuboid_sweep can be expensive —
   prefer math_job_submit + math_job_status/math_job_result (async job API)
   over blocking calls for large bounds.
6. FINITE DYNAMICS (PRNG/spectral analysis): math_finite_system_create once,
   then math_koopman_* / math_finite_fourier_compute / math_closure_search /
   math_cumulant_compute with the system_id. exact=true (default) gives
   proof-grade rational/cyclotomic values; exact=false selects the
   GPU-accelerated numeric path and downgrades trust to numeric.
   State-conditioned orbit access T^kappa(x)(x): math_conditioned_*,
   math_affine_conditioned_access, math_symbolic_conditioned_access,
   math_gf2_conditioned_access, math_gf2_predictive_closure, and the
   math_synthesize_*/math_discover_* symmetry tools.
7. PROVENANCE: every result includes derivation steps; math_derivation_trace
   reconstructs the full audit trail for any step_id.
8. MULTIMODAL PROJECTIONS: math_projection_catalog / math_projection_create
   define one typed projection shared by visualization and audio. Projection
   parameters, assumptions, evidence refs and information loss are explicit;
   >3D reductions must name their method and dimensions.
9. VISUALIZE / SONIFY: math_visualize_projection and math_sonify_projection
   consume the same projection lineage. Legacy math_visualize and math_sonify
   remain available for direct inline plots/scalar sequences. Exports are
   deterministic presentation artifacts; visible/audible patterns are not proof.
10. EXACT DISCRETE: use math_object_create + math_apply for graphs,
    combinatorial classes/generating functions, finite groups, GF(p^m),
    and Smith/Hermite module normal forms. Witnesses and certificates are
    part of the result; candidate, verified feasible, verified optimum,
    impossible, unsupported, and unknown remain distinct.
11. MULTIMODAL ARTIFACTS: math_research_artifact_create assembles stored
    visualizations + sonifications into one evidence-carrying artifact. Shared
    projection SourceRefs allow cross-modal synchronization without label
    guessing; math_export_research_artifact writes one portable HTML file.

Integers are passed as decimal strings to preserve arbitrary precision.
Ambiguous notation (e.g. implicit multiplication '1/2x') is rejected with
candidates — rewrite it explicitly and retry.""")

    @mcp.tool
    def math_capabilities(detail: str = "summary") -> dict:
        """Start with a compact domain/engine summary; detail='full' expands it.

        Use math_capability_query for paged operations and parameter schemas.
        Large full responses become retrievable, integrity-hashed resources.
        """
        return kernel.capabilities(detail=detail)

    @mcp.tool
    def math_capability_query(
        domain: str | None = None,
        object_type: str | None = None,
        input_type: str | None = None,
        output_type: str | None = None,
        operation: str | None = None,
        evidence: str | None = None,
        trust: str | None = None,
        verification_method: str | None = None,
        engine: str | None = None,
        offset: int = 0,
        limit: int = 25,
        include_schema: bool = False,
    ) -> dict:
        """Query capabilities by domain, types, operation, trust, verifier, or engine."""
        return kernel.capability_query(
            domain=domain,
            object_type=object_type,
            input_type=input_type,
            output_type=output_type,
            operation=operation,
            evidence=evidence,
            trust=trust,
            verification_method=verification_method,
            engine=engine,
            offset=offset, limit=limit, include_schema=include_schema,
        )

    @mcp.tool
    def math_result_resource_get(resource_id: str, offset: int = 0, length: int = 8192) -> dict:
        """Read a complete oversized result in byte-budgeted JSON-content pages.

        Follow next_offset, concatenate content, verify sha256, then JSON-decode.
        The delivery receipt's unknown trust is NOT the original result's trust.
        """
        return kernel.result_resource_get(resource_id, offset, length)

    @mcp.tool
    def math_object_create(object_type: str, definition: dict) -> dict:
        """Create a typed mathematical object for compositional operations.

        Supported families include transforms, complex domains/functions,
        contours, probability objects, exact graphs, combinatorial classes,
        generating functions, finite groups/rings/fields, modules, sampled signals,
        spectra, filters/designs, transfer functions, state-space systems, and
        optimization problems/certificates.
        Mathematical expressions enter by expression_id or restricted MathIR
        strings; arbitrary Python/SymPy parsing is never used. Transform
        conventions, branch metadata, graph certificates, field polynomials,
        and group tables must be explicit.
        """
        return kernel.object_create(object_type, definition).model_dump(mode="json")

    @mcp.tool
    def math_object_get(object_id: str) -> dict:
        """Fetch a typed mathematical object with its source lineage and trust."""
        return kernel.object_get(object_id).model_dump(mode="json")

    @mcp.tool
    def math_apply(
        object_id: str, operation: str, parameters: dict | None = None,
    ) -> dict:
        """Apply a typed operation and return conditions, evidence, and provenance.

        Operations dispatch from the stored object type rather than selecting a
        flat domain-specific function name. Engineering operations default to exact
        arithmetic; pass mode="numeric" explicitly for FFT/STFT, resampling,
        filter design, numerical frequency response, or numerical optimization.
        Numerical optimization is only globally certified after an independent
        exact certificate check; solver termination alone is a candidate.
        """
        return kernel.apply(object_id, operation, parameters).model_dump(mode="json")

    @mcp.tool
    def math_parse(expression: str) -> dict:
        """Parse ASCII math (e.g. 'x^2 + 2*x = 0', 'sqrt(x)/2') into MathIR.

        Returns expr_id used by all other tools. Ambiguous notation (implicit
        multiplication like '1/2x') is rejected with candidates — rewrite it
        explicitly. Supported calls: sqrt sin cos tan exp log abs factorial
        gamma binomial pi(). Relations: = != < <= > >=.
        """
        return kernel.parse(expression).model_dump(mode="json")

    @mcp.tool
    def math_parse_latex(latex: str) -> dict:
        """Parse LaTeX (e.g. '\\frac{x^2}{2} + \\sqrt{y}') into MathIR via sympy.

        Returns expr_id chainable into every other tool. Requires the
        antlr4-python3-runtime==4.11 package (pip install '.[latex]').
        """
        return kernel.parse_latex(latex).model_dump(mode="json")

    @mcp.tool
    def math_get(expr_id: str) -> dict:
        """Fetch a parsed expression's MathIR, source text, and display form."""
        return kernel.get_expression(expr_id).model_dump(mode="json")

    @mcp.tool
    def math_substitute(expr_id: str, substitutions: dict[str, str]) -> dict:
        """Replace symbols with parsed expressions, e.g. {'x': 'a + 1'}.

        Returns a new expr_id; the original is preserved.
        """
        return kernel.substitute(expr_id, substitutions).model_dump(mode="json")

    @mcp.tool
    def math_infer_structure(expr_id: str, context_id: str | None=None) -> dict:
        """Infer required algebraic capabilities, weakest structure, and side conditions."""
        return kernel.infer_structure(expr_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_codegen(expr_id: str, language: str="typescript", mode: str="generic",
                     target: str="evaluate", variable: str | None=None,
                     context_id: str | None=None) -> dict:
        """Generate verified generic code from an expression.

        language: typescript | python | rust. target: 'evaluate' (expression),
        'solve' (closed-form solver for `variable`, requires an equality), or
        'constraint' (relation checker). Emits capability-interfaced code over a
        generic algebraic structure. Only field operations are supported
        (no factorial/gamma/binomial/pi). Follow with math_verify_code.
        """
        return kernel.codegen(expr_id, language, mode, target, variable, context_id).model_dump(mode="json")

    @mcp.tool
    def math_verify_code(artifact_id: str, checks: list[str] | None=None) -> dict:
        """Verify a codegen artifact: 'typecheck' (tsc/rustc) and/or 'symbolic_roundtrip'."""
        return kernel.verify_code(artifact_id, checks).model_dump(mode="json")

    @mcp.tool
    def math_execute_code(artifact_id: str, inputs: dict[str, float]) -> dict:
        """Run a verified artifact in a sandboxed subprocess (numeric evidence only).

        Disabled unless the server is started with MATHKERNEL_ENABLE_EXECUTION=1.
        """
        return kernel.execute_code(artifact_id, inputs).model_dump(mode="json")

    @mcp.tool
    def math_yolo_settings(updates: dict | None = None) -> dict:
        """Inspect or change MATHKERNEL_* settings on the running kernel.

        Disabled unless MATHKERNEL_YOLO_MODE=true (default false). This is a
        process-wide safety gate, not a mathematical claim.

        Omit updates to list current live values plus the type schema. Pass
        updates as MATHKERNEL_* names or Settings field names, e.g.
        {"MATHKERNEL_MAX_ODE_STEPS": 200000} or {"max_ode_steps": "200000"}.
        Values are coerced to the declared type (int / float / bool / str);
        type mismatches are rejected. Null or "" clears MATHKERNEL_STORE_PATH.
        Changing timeouts/store_path/lean_binary is applied to the live
        engines immediately. MATHKERNEL_YOLO_MODE, MATHKERNEL_SKIP_LEAN_INSTALL
        and MATHKERNEL_LEAN_CACHE write the process environment only.
        """
        return kernel.yolo_settings(updates).model_dump(mode="json")

    @mcp.tool
    def math_analyze(expr_id: str, context_id: str | None=None) -> dict:
        """List symbols, node kinds, and required capabilities of an expression."""
        return kernel.analyze(expr_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_plan(expr_id: str, context_id: str | None=None, solve_for: str | None=None) -> dict:
        """Build a dependency-aware obligation DAG for a problem (heuristic routing)."""
        return kernel.plan(expr_id, context_id, solve_for=solve_for).model_dump(mode="json")

    @mcp.tool
    def math_plan_get(plan_id: str) -> dict:
        """Fetch a previously created problem plan."""
        return kernel.plan_get(plan_id).model_dump(mode="json")

    @mcp.tool
    def math_execute_plan(plan_id: str, formal: bool=True, max_steps: int=32) -> dict:
        """Execute a plan's obligations in parallel waves with conflict detection."""
        return kernel.execute_plan(plan_id, formal=formal, max_steps=max_steps).model_dump(mode="json")

    @mcp.tool
    def math_reason(expr_id: str, context_id: str | None=None, formal: bool=True, max_steps: int=32,
                    solve_for: str | None=None) -> dict:
        """Plan and execute in one call: full reasoning pipeline for an expression."""
        return kernel.reason(expr_id, context_id, formal=formal, max_steps=max_steps,
                             solve_for=solve_for).model_dump(mode="json")

    @mcp.tool
    def math_execution_get(execution_id: str) -> dict:
        """Fetch the result of a previously executed plan."""
        return kernel.execution_get(execution_id).model_dump(mode="json")

    @mcp.tool
    def math_integer_analyze(value: str, factor_limit: int=100_000) -> dict:
        """Exact analysis of an arbitrary-size integer: factorization, primality, digits."""
        return kernel.integer_analyze(value, factor_limit).model_dump(mode="json")

    @mcp.tool
    def math_integer_compute(operation: str, values: list[str], modulus: str | None=None,
                             moduli: list[str] | None=None, max_output_digits: int=100_000,
                             factor_limit: int=100_000) -> dict:
        """Exact bigint/number-theory ops: add sub mul div mod pow pow_mod gcd lcm

        Also: is_prime next_prime prev_prime factor mod_inverse crt factorial
        binomial fibonacci affine_jump. Values are decimal strings of any size.
        div is exact: it returns a reduced "p/q" rational string when the
        division is not exact (exact_rational=true in that case).

        affine_jump expects values [multiplier, increment, k] plus modulus and
        returns the exact K-step jump coefficients of the affine recurrence
        T(s) = multiplier*s + increment (mod modulus): "power" = multiplier^k
        and "translation" = increment * (1 + multiplier + ... + multiplier^(k-1)),
        both reduced mod modulus. This covers LCG/PCG-style generators where
        multiplier-1 is not invertible (the geometric sum is evaluated by
        doubling, never by division).
        """
        return kernel.integer_compute(operation, values, modulus, moduli, max_output_digits, factor_limit).model_dump(mode="json")

    @mcp.tool
    def math_integer_batch(jobs: list[dict], workers: int | None=None) -> dict:
        """Run independent integer_compute jobs in parallel worker processes.

        Each job is an object like {"operation": "is_prime", "values": ["97"]} with
        optional modulus/moduli/max_output_digits/factor_limit. Results preserve
        submission order and carry per-job ok flags.
        """
        return kernel.integer_batch(jobs, workers).model_dump(mode="json")

    @mcp.tool
    def math_collatz_sieve(n_max: int, x_min: str="1", workers: int | None=None,
                           n_min: int=1, canonical: bool=True, engine: str="auto") -> dict:
        """Refute Collatz cycle classes via the cycle closure equation.

        Exhaustively checks every halving pattern for cycles with n_min..n_max odd
        elements and all elements >= x_min, in parallel. With the default x_min="1"
        the result is self-contained; a larger x_min relies on an external
        verification floor (recorded as a side condition). With canonical=True
        (default) only rotation-canonical patterns (a_0 maximal) are enumerated;
        every cycle has such a rotation, so refutation stays complete.
        engine: auto | cuda | numba | python. For large n_max prefer
        math_job_submit so the sweep runs asynchronously.
        """
        return kernel.collatz_sieve(n_max, x_min, workers, n_min, canonical,
                                    engine).model_dump(mode="json")

    @mcp.tool
    def math_cuboid_sweep(bound: int, engine: str="auto", workers: int | None=None) -> dict:
        """Enumerate Pythagorean leg pairs a^2+b^2=c^2 with a<b<=bound (QR-prefiltered).

        engine: auto | cuda | numba | python. Building block for the perfect
        cuboid search. For large bounds prefer math_job_submit.
        """
        return kernel.cuboid_sweep(bound, engine, workers).model_dump(mode="json")

    @mcp.tool
    def math_job_submit(kind: str, params: dict | None=None) -> dict:
        """Submit a long-running sweep asynchronously. kind: collatz_sieve | cuboid_sweep.

        params must match the synchronous tool's parameters (e.g.
        {"n_max": 15, "engine": "cuda"}). Returns job_id; poll with
        math_job_status and collect with math_job_result.
        """
        return kernel.job_submit(kind, params).model_dump(mode="json")

    @mcp.tool
    def math_job_status(job_id: str) -> dict:
        """Poll a job: queued | running | done | failed, with timing."""
        return kernel.job_status(job_id).model_dump(mode="json")

    @mcp.tool
    def math_job_result(job_id: str) -> dict:
        """Collect a finished job's full MathResult payload."""
        return kernel.job_result(job_id).model_dump(mode="json")

    @mcp.tool
    def math_job_list(status: str | None=None) -> dict:
        """List retained jobs, newest first; optionally filter by status."""
        return kernel.job_list(status).model_dump(mode="json")

    @mcp.tool
    def math_context_create(domains: dict[str,str] | None=None, assumptions: list[str] | None=None) -> dict:
        """Create a context: symbol domains ('integer','real','positive',...) and assumptions.

        Domains make downstream results stronger (e.g. sqrt(x^2) simplifies to x
        when x is positive). Assumptions are parsed relations like 'x > 0'.
        """
        return kernel.create_context(domains, assumptions).model_dump(mode="json")

    @mcp.tool
    def math_context_infer(context_id: str) -> dict:
        """Derive implied symbol properties (sign, nonzero) from context assumptions."""
        return kernel.infer_context(context_id).model_dump(mode="json")

    @mcp.tool
    def math_context_check(context_id: str) -> dict:
        """Check context consistency with the Z3 SMT solver (exact)."""
        return kernel.check_context(context_id).model_dump(mode="json")

    @mcp.tool
    def math_simplify(expr_id: str, mode: str="simplify", context_id: str | None=None) -> dict:
        """Simplify an expression. mode: simplify | normal | expand | factor | cancel |
        trig | rational | normal_form. Honors context domains. Returns result_expr_id."""
        return kernel.simplify(expr_id, mode, context_id).model_dump(mode="json")

    @mcp.tool
    def math_solve(expr_id: str, variable: str, context_id: str | None=None) -> dict:
        """Solve an equation (or expression = 0) for one variable.

        Domain comes from the context (complex default; real/integer supported).
        Returns a structured solution set: finite | interval | union | conditional | ...
        """
        return kernel.solve(expr_id, variable, context_id).model_dump(mode="json")

    @mcp.tool
    def math_solve_system(expr_ids: list[str], variables: list[str], context_id: str | None=None) -> dict:
        """Solve a simultaneous system of equations for several variables.

        Each expr_id must be an equality (or expression interpreted as = 0).
        Returns explicit solutions with per-value expr_ids when representable.
        """
        return kernel.solve_system(expr_ids, variables, context_id).model_dump(mode="json")

    @mcp.tool
    def math_differentiate(expr_id: str, variable: str, order: int=1, context_id: str | None=None) -> dict:
        """Differentiate an expression: d^order/d(variable)^order. Returns result_expr_id."""
        return kernel.differentiate(expr_id, variable, order, context_id).model_dump(mode="json")

    @mcp.tool
    def math_integrate(expr_id: str, variable: str, lower: str | None=None, upper: str | None=None,
                       context_id: str | None=None) -> dict:
        """Integrate an expression. Omit bounds for the antiderivative (up to +C);

        supply both for a definite integral. Bounds may be expressions or 'oo'/'-oo'.
        """
        return kernel.integrate(expr_id, variable, lower, upper, context_id).model_dump(mode="json")

    @mcp.tool
    def math_limit(expr_id: str, variable: str, point: str, direction: str="+-",
                   context_id: str | None=None) -> dict:
        """Compute a limit as variable -> point ('oo' allowed).

        direction: '+-' two-sided, '+' from above, '-' from below.
        """
        return kernel.limit(expr_id, variable, point, direction, context_id).model_dump(mode="json")

    @mcp.tool
    def math_series(expr_id: str, variable: str, point: str="0", order: int=6,
                    context_id: str | None=None) -> dict:
        """Taylor/Laurent series expansion truncated at the given order.

        The dropped O-term is recorded as a side condition.
        """
        return kernel.series(expr_id, variable, point, order, context_id).model_dump(mode="json")

    @mcp.tool
    def math_summation(expr_id: str, variable: str, lower: str, upper: str,
                       context_id: str | None=None) -> dict:
        """Symbolic sum of an expression for variable = lower..upper ('oo' allowed)."""
        return kernel.summation(expr_id, variable, lower, upper, context_id).model_dump(mode="json")

    @mcp.tool
    def math_product(expr_id: str, variable: str, lower: str, upper: str,
                     context_id: str | None=None) -> dict:
        """Symbolic product of an expression for variable = lower..upper ('oo' allowed)."""
        return kernel.product(expr_id, variable, lower, upper, context_id).model_dump(mode="json")

    @mcp.tool
    def math_interval_evaluate(expr_id: str, bounds: dict[str,list[float|str]], dps: int=50) -> dict:
        """Certified interval enclosure of an expression over boxed symbol bounds.

        bounds: {'x': ['0', '1']}. Result is a rigorous [lo, hi] enclosure at
        dps working precision (trust: interval_certified).
        """
        return kernel.interval_evaluate(expr_id,bounds,dps).model_dump(mode="json")

    @mcp.tool
    def math_numeric_evaluate(expr_id: str, values: dict[str,str] | None=None, dps: int=50) -> dict:
        """Numeric value at dps precision; substitute symbols via values={'x': '3/2'}.

        All symbols must be bound. Not certified — use math_interval_evaluate
        for a rigorous enclosure.
        """
        return kernel.numeric_evaluate(expr_id, values, dps).model_dump(mode="json")

    @mcp.tool
    def math_matrix_create(rows: list[list[str]]) -> dict:
        """Create a matrix from rows of expression strings: [['1','2'],['3','4']].

        Cells may be symbolic. Returns matrix_id used by all matrix tools.
        """
        return kernel.matrix_create(rows).model_dump(mode="json")

    @mcp.tool
    def math_matrix_get(matrix_id: str) -> dict:
        """Fetch a matrix's cells, dimensions, and display form."""
        return kernel.matrix_get(matrix_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_det(matrix_id: str, context_id: str | None=None) -> dict:
        """Exact determinant (symbolic entries supported). Returns result_expr_id."""
        return kernel.matrix_det(matrix_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_inverse(matrix_id: str, context_id: str | None=None) -> dict:
        """Exact inverse of a square matrix; errors if singular. Returns matrix_id."""
        return kernel.matrix_inverse(matrix_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_transpose(matrix_id: str) -> dict:
        """Transpose a matrix. Returns a new matrix_id."""
        return kernel.matrix_transpose(matrix_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_multiply(left_id: str, right_id: str, context_id: str | None=None) -> dict:
        """Matrix product left x right with dimension checking. Returns matrix_id."""
        return kernel.matrix_multiply(left_id, right_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_rank(matrix_id: str, context_id: str | None=None) -> dict:
        """Exact rank of a matrix."""
        return kernel.matrix_rank(matrix_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_rref(matrix_id: str, context_id: str | None=None) -> dict:
        """Reduced row echelon form; result includes pivot_columns. Returns matrix_id."""
        return kernel.matrix_rref(matrix_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_eigenvalues(matrix_id: str, context_id: str | None=None) -> dict:
        """Exact eigenvalues with algebraic multiplicities (expr_ids when representable)."""
        return kernel.matrix_eigenvalues(matrix_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_matrix_solve(matrix_id: str, rhs_id: str, context_id: str | None=None) -> dict:
        """Solve A*x = b exactly (LU). matrix_id must be square; rhs_id a column matrix."""
        return kernel.matrix_solve(matrix_id, rhs_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_fwht(values: list[str]) -> dict:
        """Exact unnormalized Walsh-Hadamard transform of an integer vector.

        Length must be a power of two (limit MATHKERNEL_MAX_FWHT_SIZE, default 2^20).
        For a +/-1 accumulator indexed by GF(2)^n, the coefficient at index m is the
        exact Walsh correlation numerator of that mask. Satisfies fwht(fwht(v)) = n*v.
        """
        return kernel.fwht(values).model_dump(mode="json")

    @mcp.tool
    def math_set_create(elements: list[str] | None=None, name: str | None=None) -> dict:
        """Create a set and get an expr_id. Exactly one of: name (naturals,
        integers, rationals, reals, complexes, empty) or elements (expression
        strings, e.g. ["1", "2", "3"])."""
        return kernel.set_create(elements, name).model_dump(mode="json")

    @mcp.tool
    def math_set_op(op: str, set_ids: list[str]) -> dict:
        """Set algebra over stored set expr_ids: union/intersect (>=2 sets),
        difference (exactly 2), complement (set, universe). Returns a new expr_id."""
        return kernel.set_op(op, set_ids).model_dump(mode="json")

    @mcp.tool
    def math_set_membership(element: str, set_id: str) -> dict:
        """Decide element in set (exact for finite/named sets over exact elements)."""
        return kernel.set_membership(element, set_id).model_dump(mode="json")

    @mcp.tool
    def math_quantifier_check(expr_id: str, context_id: str | None=None) -> dict:
        """Validity check for a quantified proposition (parse forall(x, domain,
        body) / exists(x, domain, body) first). Returns validity=valid/invalid/
        unknown with a concrete witness or countermodel when decidable. Exact
        when Z3 decides; unknown on timeout or outside the arithmetic fragment."""
        return kernel.quantifier_check(expr_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_quantifier_eliminate(expr_id: str, context_id: str | None=None) -> dict:
        """True quantifier elimination via Z3's qe tactic (LRA/LIA): returns an
        equivalent quantifier-free MathIR formula (data.expr_id, data.formula)
        that renders and re-parses. exact trust on success; structured unknown
        with fragment classification on undecidable fragments."""
        return kernel.quantifier_eliminate(expr_id, context_id).model_dump(mode="json")

    @mcp.tool
    def math_quantifier_eliminate_batch(expr_ids: list[str],
                                        context_id: str | None=None,
                                        workers: int | None=None) -> dict:
        """Batch quantifier elimination over the process pool (QE is
        single-threaded inside Z3; parallelism is across problems)."""
        return kernel.quantifier_eliminate_batch(expr_ids, context_id, workers).model_dump(mode="json")

    @mcp.tool
    def math_poly_groebner(expr_ids: list[str], variables: list[str], order: str="lex",
                           context_id: str | None=None) -> dict:
        """Reduced Groebner basis of the ideal generated by the given polynomial
        expr_ids. Exact (rational coefficients). order: lex/grlex/grevlex."""
        return kernel.poly_groebner(expr_ids, variables, order, context_id).model_dump(mode="json")

    @mcp.tool
    def math_poly_divide(dividend_id: str, divisor_ids: list[str], variables: list[str],
                         order: str="lex", context_id: str | None=None) -> dict:
        """Exact multivariate polynomial division: quotients and remainder."""
        return kernel.poly_divide(dividend_id, divisor_ids, variables, order, context_id).model_dump(mode="json")

    @mcp.tool
    def math_poly_resultant(a_id: str, b_id: str, variable: str,
                            context_id: str | None=None) -> dict:
        """Exact resultant of two polynomials with respect to variable."""
        return kernel.poly_resultant(a_id, b_id, variable, context_id).model_dump(mode="json")

    @mcp.tool
    def math_poly_discriminant(expr_id: str, variable: str, context_id: str | None=None) -> dict:
        """Exact discriminant of a univariate polynomial."""
        return kernel.poly_discriminant(expr_id, variable, context_id).model_dump(mode="json")

    @mcp.tool
    def math_poly_factor(expr_id: str, extension: str | None=None,
                         context_id: str | None=None) -> dict:
        """Exact factorization over ZZ/QQ, optionally over an algebraic extension
        (e.g. extension="sqrt(2)")."""
        return kernel.poly_factor(expr_id, extension, context_id).model_dump(mode="json")

    @mcp.tool
    def math_ideal_membership(expr_id: str, generator_ids: list[str], variables: list[str],
                              order: str="lex", context_id: str | None=None) -> dict:
        """Exact ideal membership test: is expr_id in <generator_ids>? Uses the
        Groebner remainder test; returns member=true/false plus the remainder."""
        return kernel.ideal_membership(expr_id, generator_ids, variables, order, context_id).model_dump(mode="json")

    @mcp.tool
    def math_poly_groebner_batch(jobs: list[dict], workers: int | None=None) -> dict:
        """Independent Groebner bases across worker processes. Each job:
        {"polys": [str, ...], "variables": [str, ...], "order": "lex"?}."""
        return kernel.poly_groebner_batch(jobs, workers).model_dump(mode="json")

    @mcp.tool
    def math_prob_rv_create(values: list[str], probabilities: list[str]) -> dict:
        """Create a discrete random variable with exact rational probabilities
        (strings like "1/3"). Probabilities must sum to 1."""
        return kernel.prob_rv_create(values, probabilities).model_dump(mode="json")

    @mcp.tool
    def math_prob_expectation(rv_id: str, power: int=1) -> dict:
        """Exact E[X^power] for a discrete rational RV."""
        return kernel.prob_expectation(rv_id, power).model_dump(mode="json")

    @mcp.tool
    def math_prob_variance(rv_id: str) -> dict:
        """Exact variance of a discrete rational RV."""
        return kernel.prob_variance(rv_id).model_dump(mode="json")

    @mcp.tool
    def math_prob_covariance(x_values: list[str], y_values: list[str],
                             joint_probabilities: list[str]) -> dict:
        """Exact covariance from a joint pmf over paired outcomes."""
        return kernel.prob_covariance(x_values, y_values, joint_probabilities).model_dump(mode="json")

    @mcp.tool
    def math_prob_bayes(prior: list[str], likelihood: list[str]) -> dict:
        """Exact Bayes posterior: posterior_i proportional to likelihood_i * prior_i."""
        return kernel.prob_bayes(prior, likelihood).model_dump(mode="json")

    @mcp.tool
    def math_prob_markov_stationary(matrix_id: str) -> dict:
        """Exact stationary distribution of a rational row-stochastic transition
        matrix (create it with math_matrix_create)."""
        return kernel.prob_markov_stationary(matrix_id).model_dump(mode="json")

    @mcp.tool
    def math_prob_markov_hitting_time(matrix_id: str, targets: list[int]) -> dict:
        """Exact expected hitting times to the target state set (null = never reaches)."""
        return kernel.prob_markov_hitting_time(matrix_id, targets).model_dump(mode="json")

    @mcp.tool
    def math_prob_sample(rv_id: str, n: int, seed: int | None=None) -> dict:
        """Draw n samples from a discrete RV (numeric evidence; seeded)."""
        return kernel.prob_sample(rv_id, n, seed).model_dump(mode="json")

    @mcp.tool
    def math_prob_distribution(distribution: str, parameters: list[str], query: str,
                               point: str | None=None) -> dict:
        """Continuous/symbolic distributions via sympy.stats (symbolic trust).
        distribution: normal/exponential/uniform/binomial/poisson/bernoulli/
        geometric/beta/gamma/cauchy. query: expectation/variance/std/density/cdf
        (density and cdf require point)."""
        return kernel.prob_distribution(distribution, parameters, query, point).model_dump(mode="json")

    @mcp.tool
    def math_stats_moments(values: list[str], max_order: int=4) -> dict:
        """Exact sample moments: mean, population/sample variance, central moments."""
        return kernel.stats_moments(values, max_order).model_dump(mode="json")

    @mcp.tool
    def math_stats_order(values: list[str]) -> dict:
        """Exact order statistics: sorted sample, min/max, median, quartiles."""
        return kernel.stats_order(values).model_dump(mode="json")

    @mcp.tool
    def math_stats_regression(x_values: list[str], y_values: list[str]) -> dict:
        """Exact rational least-squares regression: slope, intercept, R^2."""
        return kernel.stats_regression(x_values, y_values).model_dump(mode="json")

    @mcp.tool
    def math_stats_correlation(x_values: list[str], y_values: list[str]) -> dict:
        """Pearson correlation: exact r^2 and covariance; r is high-precision
        numeric, so overall trust is numeric_high_precision."""
        return kernel.stats_correlation(x_values, y_values).model_dump(mode="json")

    @mcp.tool
    def math_stats_ttest(values: list[str], mu0: str) -> dict:
        """One-sample t-test against mu0 (mpmath; numeric_high_precision, not proof)."""
        return kernel.stats_ttest(values, mu0).model_dump(mode="json")

    @mcp.tool
    def math_stats_chi2(observed: list[str], expected: list[str] | None=None) -> dict:
        """Pearson chi-square goodness-of-fit (numeric_high_precision)."""
        return kernel.stats_chi2(observed, expected).model_dump(mode="json")

    @mcp.tool
    def math_stats_confidence_interval(values: list[str], confidence: str="0.95") -> dict:
        """t-based confidence interval for the mean (numeric_high_precision)."""
        return kernel.stats_confidence_interval(values, confidence).model_dump(mode="json")

    @mcp.tool
    def math_stats_batch_moments(columns: list[list[str]], workers: int | None=None,
                                 numeric: bool=False) -> dict:
        """Moments for many sample columns across the process pool. numeric=True
        takes the vectorized float64 path (trust numeric)."""
        return kernel.stats_batch_moments(columns, workers, numeric).model_dump(mode="json")

    @mcp.tool
    def math_tensor_create(shape: list[int], entries: dict[str, str]) -> dict:
        """Create a sparse tensor. entries maps "i,j,k" index strings to exact
        rationals ("2", "1/3") or float literals ("0.5" -> numeric trust)."""
        return kernel.tensor_create(shape, entries).model_dump(mode="json")

    @mcp.tool
    def math_tensor_get(tensor_id: str) -> dict:
        """Fetch a tensor: full entries when small, shape/nnz summary when large."""
        return kernel.tensor_get(tensor_id).model_dump(mode="json")

    @mcp.tool
    def math_tensor_contract(spec: str, tensor_ids: list[str], exact: bool=True) -> dict:
        """Einstein-style contraction, e.g. "ij,jk->ik". exact=True uses exact
        rational arithmetic; exact=False dispatches to njit/CuPy sparse tiers
        (trust numeric)."""
        return kernel.tensor_contract(spec, tensor_ids, exact).model_dump(mode="json")

    @mcp.tool
    def math_tensor_solve(a_id: str, b_id: str) -> dict:
        """Exact sparse solve A x = b over the rationals (sparse Gauss-Jordan).
        A: rank-2 square tensor; b: rank-1 or rank-2."""
        return kernel.tensor_solve(a_id, b_id).model_dump(mode="json")

    @mcp.tool
    def math_root_find(expr_id: str, variable: str, a: str | None=None,
                       b: str | None=None, x0: str | None=None,
                       certified: bool=False, dps: int=50, fast: bool=False) -> dict:
        """Find a root of a univariate expression. certified=True: rigorous
        interval isolation (interval_certified). Default: arbitrary-precision
        Ridder/secant (numeric_high_precision). fast=True: float64 Brent
        (numeric). Bounds accept expressions like "pi" or "1/3"."""
        return kernel.root_find(expr_id, variable, a, b, x0, certified, dps, fast).model_dump(mode="json")

    @mcp.tool
    def math_root_scan(expr_id: str, variable: str, a: str, b: str,
                       intervals: int=64, dps: int=50, workers: int | None=None) -> dict:
        """Parallel scan of [a, b] for all sign-changing roots (process pool,
        numeric_high_precision)."""
        return kernel.root_scan(expr_id, variable, a, b, intervals, dps, workers).model_dump(mode="json")

    @mcp.tool
    def math_quadrature(expr_id: str, variable: str, a: str, b: str,
                        dps: int=50, cross_check: bool=True) -> dict:
        """Definite integral via tanh-sinh at arbitrary precision, cross-checked
        with Gauss-Legendre; disagreement returns status=conflict.
        numeric_high_precision, not proof."""
        return kernel.quadrature(expr_id, variable, a, b, dps, cross_check).model_dump(mode="json")

    @mcp.tool
    def math_ode_solve(rhs_id: str, y_var: str="y", x_var: str="x",
                       ics: dict[str, str] | None=None) -> dict:
        """Symbolic dy/dx = rhs(x, y) via sympy.dsolve with ODE classification
        (symbolic trust). In the rhs, the bare symbol y means y(x)."""
        return kernel.ode_solve(rhs_id, y_var, x_var, ics).model_dump(mode="json")

    @mcp.tool
    def math_ode_solve_numeric(rhs_ids: list[str], t_span: list[str], y0: list[str],
                               tol: float | None=None, dps: int=50, fast: bool=False) -> dict:
        """Numeric IVP for a first-order system: rhs_ids[i] is dy_i/dt with state
        variables named y0, y1, ... Default adaptive RK45 on mpmath
        (numeric_high_precision); fast=True is float64 RK4 (numeric)."""
        return kernel.ode_solve_numeric(rhs_ids, t_span, y0, tol, dps, fast).model_dump(mode="json")

    @mcp.tool
    def math_ode_ensemble(rhs_ids: list[str], t_span: list[str], y0s: list[list[float]],
                          steps: int=1000, prefer_gpu: bool=True,
                          workers: int | None=None) -> dict:
        """Many-trajectory IVP batch: GPU RawKernel (one thread per trajectory)
        when CUDA works, else a process pool of float64 RK4. Numeric trust."""
        return kernel.ode_ensemble(rhs_ids, t_span, y0s, steps, prefer_gpu, workers).model_dump(mode="json")

    @mcp.tool
    def math_pde_heat_1d(u0: list[float], alpha: float, dx: float, dt: float,
                         steps: int, prefer_gpu: bool=True) -> dict:
        """1D heat equation u_t = alpha*u_xx via explicit FTCS finite differences
        (numeric evidence only). Enforces the r <= 1/2 stability condition."""
        return kernel.pde_heat_1d(u0, alpha, dx, dt, steps, prefer_gpu).model_dump(mode="json")

    @mcp.tool
    def math_pde_heat_2d(u0: list[list[float]], alpha: float, dx: float,
                         dt: float, steps: int, prefer_gpu: bool=True) -> dict:
        """2D heat equation u_t = alpha*(u_xx+u_yy), FTCS, Dirichlet-zero
        boundaries. Stability r <= 1/4 enforced. Numeric evidence only."""
        return kernel.pde_heat_2d(u0, alpha, dx, dt, steps, prefer_gpu).model_dump(mode="json")

    @mcp.tool
    def math_pde_wave_1d(u0: list[float], v0: list[float], c: float, dx: float,
                        dt: float, steps: int, prefer_gpu: bool=True) -> dict:
        """1D wave equation u_tt = c^2 u_xx, leapfrog central differences.
        CFL c*dt/dx <= 1 enforced. Numeric evidence only."""
        return kernel.pde_wave_1d(u0, v0, c, dx, dt, steps, prefer_gpu).model_dump(mode="json")

    @mcp.tool
    def math_pde_advect_1d(u0: list[float], c: float, dx: float, dt: float,
                           steps: int, prefer_gpu: bool=True) -> dict:
        """1D advection u_t + c u_x = 0, first-order upwind (c >= 0, CFL
        enforced). First-order upwind is numerically diffusive."""
        return kernel.pde_advect_1d(u0, c, dx, dt, steps, prefer_gpu).model_dump(mode="json")

    @mcp.tool
    def math_pde_ensemble(u0: list[list[float]], alphas: list[float],
                          dx: float, dt: float, steps: int,
                          prefer_gpu: bool=True, workers: int | None=None) -> dict:
        """2D heat parameter sweep over diffusivities: batched CuPy stencil on
        GPU, process pool on CPU. Numeric evidence only."""
        return kernel.pde_ensemble(u0, alphas, dx, dt, steps, prefer_gpu, workers).model_dump(mode="json")

    @mcp.tool
    def math_pde_mol_heat(u0: list[float], alpha: float, dx: float,
                          t1: float, steps: int=1000) -> dict:
        """Method of lines: semidiscretized 1D heat equation integrated by the
        RK4 float64 ODE solver. Numeric evidence only."""
        return kernel.pde_mol_heat(u0, alpha, dx, t1, steps).model_dump(mode="json")

    @mcp.tool
    def math_optimize_critical_points(expr_id: str, variables: list[str],
                                      context_id: str | None=None) -> dict:
        """Symbolic critical points: solves grad f = 0 (symbolic trust)."""
        return kernel.optimize_critical_points(expr_id, variables, context_id).model_dump(mode="json")

    @mcp.tool
    def math_optimize_kkt(objective_id: str, constraint_ids: list[str],
                          variables: list[str], context_id: str | None=None) -> dict:
        """KKT conditions for min f s.t. g_i(x) <= 0: stationarity, primal/dual
        feasibility, complementary slackness (symbolic trust)."""
        return kernel.optimize_kkt(objective_id, constraint_ids, variables, context_id).model_dump(mode="json")

    @mcp.tool
    def math_lp_solve(c: list[str], a: list[list[str]], b: list[str],
                      exact: bool=True) -> dict:
        """Linear program: max c^T x s.t. A x <= b, x >= 0, b >= 0.
        exact=True: rational simplex with Bland's rule (exact); exact=False:
        njit float64 tableau (numeric)."""
        return kernel.lp_solve(c, a, b, exact).model_dump(mode="json")

    @mcp.tool
    def math_optimize_minimize(expr_id: str, variables: list[str],
                               start: list[float], tol: float | None=None,
                               max_iter: int | None=None) -> dict:
        """Local minimization via Nelder-Mead (float64, numeric trust)."""
        return kernel.optimize_minimize(expr_id, variables, start, tol, max_iter).model_dump(mode="json")

    @mcp.tool
    def math_optimize_multistart(expr_id: str, variables: list[str],
                                 starts: list[list[float]], workers: int | None=None) -> dict:
        """Multi-start Nelder-Mead across the process pool; returns the best
        result plus all endpoints (numeric trust)."""
        return kernel.optimize_multistart(expr_id, variables, starts, workers).model_dump(mode="json")

    @mcp.tool
    def math_unit_check(expr_id: str, units: dict[str, str] | None=None) -> dict:
        """Dimensional analysis: assign units to free symbols
        ({"v": "m/s", "t": "s"}) and get the SI dimension of the expression.
        Inconsistent dimensions are hard errors. Exact trust."""
        return kernel.unit_check(expr_id, units).model_dump(mode="json")

    @mcp.tool
    def math_unit_convert(value: str, from_unit: str, to_unit: str) -> dict:
        """Exact rational unit conversion, e.g. ("36", "km/h", "m/s") -> 10.
        Dimensional mismatch is an error. Exact trust."""
        return kernel.unit_convert(value, from_unit, to_unit).model_dump(mode="json")

    @mcp.tool
    def math_unit_simplify(unit: str) -> dict:
        """Reduce a unit expression ("kg*m/s^2") to its SI dimension vector and
        exact scale factor. Exact trust."""
        return kernel.unit_simplify(unit).model_dump(mode="json")

    @mcp.tool
    def math_store_status() -> dict:
        """Whether SQLite persistence is active (MATHKERNEL_STORE_PATH)."""
        return kernel.store_status().model_dump(mode="json")

    @mcp.tool
    def math_replay(step_id: str) -> dict:
        """Deterministic replay of a derivation DAG from the persistent store:
        topological provenance chain with integrity validation."""
        return kernel.replay(step_id).model_dump(mode="json")

    @mcp.tool
    def math_fuzz_differential(n: int=100, variables: list[str] | None=None,
                               depth: int=3, samples: int=8, seed: int=0,
                               workers: int | None=None) -> dict:
        """Cross-engine differential fuzzing of the numeric fragment
        (mpmath high-precision vs float64). status=conflict means a tier
        disagreement was found — inspect data.disagreements."""
        return kernel.fuzz_differential(n, variables, depth, samples, seed, workers).model_dump(mode="json")

    @mcp.tool
    def math_certified_enclose(expr_id: str, variable: str, lo: str, hi: str) -> dict:
        """Rigorous interval enclosure of a univariate expression over [lo, hi]:
        Arb ball arithmetic when python-flint is installed, else mpmath.iv.
        interval_certified trust."""
        return kernel.certified_enclose(expr_id, variable, lo, hi).model_dump(mode="json")

    @mcp.tool
    def math_prove(expr_id: str, context_id: str | None=None, formal: bool=True) -> dict:
        """General theorem proving: classifies the fragment, races a portfolio
        of Z3 encodings, and (for supported relation goals) emits a Lean
        certificate. Trust: formal only with a checked certificate, exact for
        decisive SMT, unknown on timeout. status: valid | not_valid | unknown."""
        return kernel.prove(expr_id, context_id, formal).model_dump(mode="json")

    @mcp.tool
    def math_prove_batch(expr_ids: list[str], context_id: str | None=None,
                         workers: int | None=None) -> dict:
        """Prove many statements in parallel (process pool, SMT tier)."""
        return kernel.prove_batch(expr_ids, context_id, workers).model_dump(mode="json")

    @mcp.tool
    def math_prove_replay(cert_id: str) -> dict:
        """Re-check a stored Lean certificate (requires MATHKERNEL_STORE_PATH).
        formal trust when the replay proves."""
        return kernel.prove_replay(cert_id).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_create(degree: int, reduction: str) -> dict:
        """Create GF(2^degree) with modulus x^degree + reduction (hex string).

        Irreducibility is verified with Rabin's test; a reducible modulus is
        rejected because inverses/roots would be unsound. Returns field_id.
        """
        return kernel.gf2m_create(degree, reduction).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_from_transition(columns: list[str]) -> dict:
        """Build GF(2^m) from the columns of a GF(2)-linear transition (hex strings).

        Derives the transition's minimal polynomial via the dual-orbit cyclic
        basis and uses it as the field modulus (irreducibility verified).
        This is the construction behind LFSR/xoroshiro jump and closure analysis.
        """
        return kernel.gf2m_from_transition(columns).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_compute(field_id: str, operation: str, values: list[str],
                          exponent: str | None=None) -> dict:
        """Exact GF(2^m) arithmetic. operation: add sub mul div pow inv sqrt trace
        quadratic_roots. Values are hex strings; pow takes a hex exponent.
        quadratic_roots expects [c0, c1, c2] solving c0 + c1*r + c2*r^2 = 0.
        """
        return kernel.gf2m_compute(field_id, operation, values, exponent).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_coords(field_id: str, row: str) -> dict:
        """Map a state-row bit vector to field coordinates (transition-derived fields)."""
        return kernel.gf2m_coords(field_id, row).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_root_jump_rows(field_id: str, root: str) -> dict:
        """Rows of the state-jump action induced by a field element (hex)."""
        return kernel.gf2m_root_jump_rows(field_id, root).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_closure_roots(field_id: str, row_sets: list[list[str]]) -> dict:
        """Solve closure equations for state-row sets: 2 rows -> linear c0 + c1*r = 0,

        3 rows -> quadratic c0 + c1*r + c2*r^2 = 0. Returns roots with residuals;
        closure_valid means every residual is zero. Trivial roots 0 and 1 excluded.
        """
        return kernel.gf2m_closure_roots(field_id, row_sets).model_dump(mode="json")

    @mcp.tool
    def math_gf2m_jump_rows(field_id: str, k: str) -> dict:
        """Rows of the K-step jump T^K of the transition (= action of alpha^K).

        k may be decimal or 0x-prefixed hex; it is reduced mod 2^m - 1.
        Applying these rows to a state advances it by K generator steps exactly.
        """
        return kernel.gf2m_jump_rows(field_id, k).model_dump(mode="json")

    @mcp.tool
    def math_gf2_rank(rows: list[str], width: int) -> dict:
        """Rank and nullity of a GF(2) matrix given as row parity masks (hex)."""
        return kernel.gf2_rank(rows, width).model_dump(mode="json")

    @mcp.tool
    def math_gf2_nullspace(rows: list[str], width: int, side: str="right") -> dict:
        """Basis of the right nullspace {v: Mv = 0} or left nullspace {v: vM = 0}

        of a GF(2) matrix given as row parity masks (hex). side: right|left.
        """
        return kernel.gf2_nullspace(rows, width, side).model_dump(mode="json")

    @mcp.tool
    def math_gf2_carryfree_cols(c: str, width: int) -> dict:
        """Columns of the carry-free multiply-by-c map on width-bit words.

        P_c(x) = XOR over set bits j of c of (x << j): the GF(2)-linear analogue
        of integer multiplication by c, used to lift integer plane normals
        (e.g. 3x + 16y - 22z) to carry-free GF(2) operators.
        """
        return kernel.gf2_carryfree_cols(c, width).model_dump(mode="json")

    @mcp.tool
    def math_gf2_minpoly(bits_hex: str, nbits: int, verify_from: int | None=None) -> dict:
        """Minimal (connection) polynomial of a GF(2) bit sequence.

        Berlekamp-Massey over the packed sequence s_0..s_{nbits-1} (s_i = bit i
        of bits_hex). Returns the connection polynomial C(x) (hex, bit j =
        coefficient of x^j), its degree (linear complexity) and Hamming weight.
        The recurrence s_n = sum_{j=1..L} c_j s_{n-j} is exact from position L
        onward; pass verify_from to count violations on the tail (0 = exact).
        Classic use: recovering the characteristic polynomial weight (N1) of an
        F2-linear generator from its MSB stream (e.g. MT19937 -> degree 19937,
        weight 135).
        """
        return kernel.gf2_minpoly(bits_hex, nbits, verify_from).model_dump(mode="json")

    @mcp.tool
    def math_prove_equivalence(left: str, right: str, context_id: str | None=None, formal: bool=True) -> dict:
        """Prove or refute that two expressions are equivalent.

        Collects independent evidence from sympy (symbolic), z3 (exact SMT), and
        lean (formal certificate when formal=true). Status: verified | refuted |
        unknown, with per-engine evidence and the highest trust level achieved.
        """
        return kernel.prove_equivalence(left,right,context_id,formal).model_dump(mode="json")

    @mcp.tool
    def math_counterexample(left: str, right: str, context_id: str | None=None) -> dict:
        """Search for a concrete counterexample to left = right via Z3 (exact)."""
        return kernel.counterexample(left,right,context_id).model_dump(mode="json")

    @mcp.tool
    def math_derivation_get(step_id: str) -> dict:
        """Fetch one derivation step (operation, inputs, output, trust, engine)."""
        return kernel.derivation_get(step_id).model_dump(mode="json")

    @mcp.tool
    def math_derivation_trace(step_id: str) -> dict:
        """Trace the full provenance DAG behind a derivation step."""
        return kernel.derivation_trace(step_id).model_dump(mode="json")

    @mcp.tool
    def math_cumulant_compute(op: str, order: int, values: dict | None=None,
                              columns: list | None=None) -> dict:
        """Algebraic joint cumulants via the set-partition lattice (exact).

        op='cumulant': connected cumulant from raw block moments. op='moment':
        raw moment from connected block cumulants (inverse). op='samples':
        cumulant directly from sample columns. For 'cumulant'/'moment', values
        maps subset keys to exact rational strings — the key lists coordinates,
        e.g. '0,1,3' -> E[Z0*Z1*Z3] (or Cum); all 2^order-1 nonempty subsets
        are required. No complex conjugation is inserted in any argument.
        """
        return kernel.cumulant_compute(op, order, values, columns).model_dump(mode="json")

    @mcp.tool
    def math_finite_system_create(mu, transition: list[int],
                                  observation: list[int] | None=None) -> dict:
        """Create a finite partially observed dynamical system (X, mu, T, O).

        States are indices 0..n-1; transition[x] is the successor of x;
        observation[x] is the output index of x (default: identity). mu is a
        list of exact rational strings ('1/8') or the string 'uniform'.
        Returns system_id plus stationarity of the measure. Limit:
        max_finite_states (default 4096).
        """
        return kernel.finite_system_create(mu, transition, observation).model_dump(mode="json")

    @mcp.tool
    def math_koopman_matrix(system_id: str, basis: dict, exact: bool=True) -> dict:
        """Koopman transport matrix Q[b][a] = <U psi_a, psi_b> in a state basis.

        basis = {'kind': 'walsh', 'r': r} for GF(2)^r (states 0..2^r-1) or
        {'kind': 'cyclic', 'm': m} for Z_M characters. exact=true (default):
        rational for Walsh, cyclotomic for cyclic — zeros are proofs.
        exact=false: complex128 numeric path (GPU via CuPy when available),
        much faster for large systems, trust downgraded to numeric.
        Under an invariant measure Q is unitary.
        """
        return kernel.koopman_matrix(system_id, basis, exact).model_dump(mode="json")

    @mcp.tool
    def math_koopman_transfer(system_id: str, output_functions: list, basis: dict,
                              exact: bool=True) -> dict:
        """Observation-transfer matrix C[h][a] = <phi_h ∘ O, psi_a>.

        output_functions[h][y] gives phi_h on the output set (exact rational
        strings, or {'re','im'} for complex). C depends on the observation and
        output dictionary only — not on the dynamics or lag tuple.
        exact=false selects the vectorized numeric path (CuPy GPU if present).
        """
        return kernel.koopman_transfer(system_id, output_functions, basis,
                                       exact).model_dump(mode="json")

    @mcp.tool
    def math_koopman_visibility(system_id: str, basis: dict, exact: bool=True) -> dict:
        """Mode visibility rho_O(a) = ||P_O psi_a||^2 for every basis mode.

        Exact in [0,1]: 0 means the mode is invisible to the observation
        (orthogonal to the observation subspace), 1 means fully determined by
        it. Lists invisible and fully visible mode indices explicitly.
        exact=false: numeric path (1e-9 zero/full tolerance), GPU-accelerated.
        """
        return kernel.koopman_visibility(system_id, basis, exact).model_dump(mode="json")

    @mcp.tool
    def math_koopman_lagged(system_id: str, basis: dict, tau: list[int],
                            alphas: list[int], connected: bool=True,
                            exact: bool=True) -> dict:
        """Lagged state tensor entry J_tau(alpha), raw or connected.

        J = E[prod_j psi_{a_j}(T^{tau_j} S)]; connected=true applies exact
        cumulant partition subtraction. Depends on the dynamics and lag
        geometry only — not on the observation. exact=false: vectorized
        numeric enumeration (CuPy GPU if present), trust numeric.
        """
        return kernel.koopman_lagged(system_id, basis, tau, alphas, connected,
                                     exact).model_dump(mode="json")

    @mcp.tool
    def math_koopman_observed(system_id: str, output_functions: list, h_tuple: list[int],
                              tau: list[int], connected: bool=True,
                              exact: bool=True) -> dict:
        """Connected observed statistic K_h^(d)(tau) by exact enumeration.

        Z_j = phi_{h_j}(O(T^{tau_j} S)); the joint cumulant over the support.
        This is the ground-truth scalar that the basis contraction
        sum_alpha (prod_j C[h_j][a_j]) * Jc_tau(alpha) reproduces exactly.
        exact=false: vectorized numeric enumeration (CuPy GPU if present).
        """
        return kernel.koopman_observed(system_id, output_functions, h_tuple, tau,
                                       connected, exact).model_dump(mode="json")

    @mcp.tool
    def math_koopman_diagnostics(system_id: str, basis: dict, exact: bool=True) -> dict:
        """Spectral spreading of Koopman transport: per-column IPR and entropy.

        IPR = sum_b |Q[b][a]|^4 is exact (1 = monomial transport, e.g. affine
        dynamics in a character basis); the entropy is numeric. Reports
        stationarity of the system measure alongside. exact=false computes Q
        and the diagnostics on the numeric (GPU-accelerated) path.
        """
        return kernel.koopman_diagnostics(system_id, basis, exact).model_dump(mode="json")

    @mcp.tool
    def math_finite_fourier_compute(op: str, params: dict) -> dict:
        """Exact Fourier analysis on Z_M (cyclotomic arithmetic).

        op='transfer': T_F(h,k), the state-character coefficients of the
        pulled-back output character chi_h ∘ F — params {F: [int], N, h}.
        op='two_point': B_m(K) for an affine K-step map T^K(x)=A_K*x+B_K —
        params {F, N, m, a_k, b_k}. op='measure': mu_hat(k) for a measure on
        Z_M — params {mu: [rational strings]}. op='dft': conjugated DFT —
        params {values: [...]}. op='orbit_correction': nonzero-orbit average
        from closure sums — params {closed_sum, total_sum, orbit_size}.
        """
        return kernel.finite_fourier_compute(op, **params).model_dump(mode="json")

    @mcp.tool
    def math_conditioned_access_solve(system_id: str, target: list[int]) -> dict:
        """Solve the exact least state-dependent lags k(x) with T^k(x)=target[x]."""
        return kernel.conditioned_access_solve(system_id, target).model_dump(mode="json")

    @mcp.tool
    def math_conditioned_symmetry_access(system_id: str, transform: list[int]) -> dict:
        """Prove O(S(x))=O(x) and convert the symmetry S into exact orbit-access lags."""
        return kernel.conditioned_symmetry_access(system_id, transform).model_dump(mode="json")

    @mcp.tool
    def math_conditioned_access_compose(system_id: str, first: list[int],
                                        second: list[int]) -> dict:
        """Compose state-conditioned access maps with the exact access-cocycle law."""
        return kernel.conditioned_access_compose(system_id, first, second).model_dump(mode="json")

    @mcp.tool
    def math_conditioned_closure(system_id: str, accesses: list[list[int]],
                                 coefficients: list[int], modulus: int,
                                 constant: int=0) -> dict:
        """Prove sum h_j O(T^kappa_j(x)x)=constant (mod N) over every finite state."""
        return kernel.conditioned_closure(system_id, accesses, coefficients,
                                          modulus, constant).model_dump(mode="json")

    @mcp.tool
    def math_symbolic_conditioned_access(modulus: int, increment: int,
                                         target_kind: str, target_constant: int,
                                         variable: str="x") -> dict:
        """Derive compact exact symbolic kappa(x) for affine cyclic dynamics."""
        return kernel.symbolic_conditioned_access(modulus,increment,target_kind,
                                                  target_constant,variable).model_dump(mode="json")

    @mcp.tool
    def math_synthesize_gf2_vector_conditioned_access(outputs: list[dict],
                                                       word_bits: int, constants: list[int],
                                                       columns: list[int], state_words: list[int],
                                                       max_lag: int, probe_samples: int=2048,
                                                       top_k: int=64) -> dict:
        """Synthesize two-word symmetries and solve exact GF(2) conditioned accesses."""
        return kernel.synthesize_gf2_vector_conditioned_access(
            outputs,word_bits,constants,columns,state_words,max_lag,
            probe_samples,top_k).model_dump(mode="json")

    @mcp.tool
    def math_gf2_predictive_closure(columns: list[int], lag: int,
                                    sparse_term_limit: int=16) -> dict:
        """Derive and exactly verify a GF(2) predictive closure at a chosen lag."""
        return kernel.gf2_predictive_closure(
            columns,lag,sparse_term_limit).model_dump(mode="json")

    @mcp.tool
    def math_gf2_conditioned_access(columns: list[int], state: int, target: int,
                                    max_lag: int) -> dict:
        """Exactly solve a bounded state-conditioned orbit access for cyclic GF(2) dynamics."""
        return kernel.gf2_conditioned_access(columns,state,target,max_lag).model_dump(mode="json")

    @mcp.tool
    def math_synthesize_conditioned_closures(expression: dict, word_bits: int,
                                             increment: int, constants: list[int],
                                             variable: str="x", max_depth: int=2,
                                             probe_samples: int=1024,
                                             proof_top_k: int=64) -> dict:
        """Synthesize/rank small word transformations and exactly prove surviving closures."""
        return kernel.synthesize_conditioned_closures(
            expression,word_bits,increment,constants,variable,max_depth,
            probe_samples,proof_top_k).model_dump(mode="json")

    @mcp.tool
    def math_discover_structural_conditioned_closure(expression: dict,
                                                      word_bits: int,
                                                      increment: int,
                                                      constants: list[int],
                                                      variable: str="x") -> dict:
        """Discover exact expression symmetries and derive conditioned orbit closures."""
        return kernel.discover_structural_conditioned_closure(
            expression,word_bits,increment,constants,variable).model_dump(mode="json")

    @mcp.tool
    def math_discover_factor_swap_conditioned_closure(word_bits: int, increment: int,
                                                       xor_constant: int) -> dict:
        """Discover/prove fold(x*(x xor C)) symmetry and its exact conditioned lag."""
        return kernel.discover_factor_swap_conditioned_closure(
            word_bits,increment,xor_constant).model_dump(mode="json")

    @mcp.tool
    def math_affine_conditioned_access(modulus: int, increment: int,
                                       state: int, target_state: int) -> dict:
        """Exact arbitrary-integer solution of x+k*A=target (mod M), gcd(A,M)=1."""
        return kernel.affine_conditioned_access(modulus, increment, state,
                                                target_state).model_dump(mode="json")

    @mcp.tool
    def math_projection_catalog() -> dict:
        """List the canonical multimodal projection object families.

        Visualization and sonification consume these same typed projections,
        so mathematical lineage, assumptions, explicit projection choices and
        declared information loss stay synchronized across modalities.
        """
        return kernel.projection_catalog().model_dump(mode="json")

    @mcp.tool
    def math_projection_create(kind: str | None = None,
                               payload: dict | None = None,
                               title: str | None = None,
                               source_object_id: str | None = None,
                               source_ref: dict | None = None,
                               trust: str = "unknown",
                               parameters: dict | None = None,
                               coordinate_names: list[str] | None = None,
                               units: dict[str, str] | None = None,
                               assumptions: list[str] | None = None,
                               evidence_refs: list[str] | None = None,
                               information_loss: list[str] | None = None,
                               information_loss_notes: list[str] | None = None,
                               metadata: dict | None = None) -> dict:
        """Create a canonical visualization/audio projection object.

        Pass kind+payload explicitly, or only source_object_id (a typed object
        from math_object_create) to let the registered domain adapter choose
        kind and payload automatically; adapter sampling/reduction choices are
        declared in the resulting projection's parameters/information_loss.

        Important: higher-dimensional projections (>3D) must explicitly name
        the projection method, input/output dimensions, and declare
        information_loss=["projection"].  Never hide slicing, aggregation,
        flattening or dimensionality reduction in a renderer.
        """
        return kernel.projection_create(
            kind, payload, title=title, source_object_id=source_object_id,
            source_ref=source_ref, trust=trust, parameters=parameters,
            coordinate_names=coordinate_names, units=units,
            assumptions=assumptions, evidence_refs=evidence_refs,
            information_loss=information_loss,
            information_loss_notes=information_loss_notes,
            metadata=metadata).model_dump(mode="json")

    @mcp.tool
    def math_projection_describe(projection_id: str) -> dict:
        """Return the complete projection, including lineage and information loss."""
        return kernel.projection_describe(projection_id).model_dump(mode="json")

    @mcp.tool
    def math_visualize_projection(projection_id: str,
                                  title: str | None = None) -> dict:
        """Render any canonical projection through mathkernel-viz."""
        return kernel.viz_projection(projection_id, title).model_dump(mode="json")

    @mcp.tool
    def math_sonify_projection(projection_id: str, mode: str = "auto",
                               title: str | None = None,
                               options: dict | None = None) -> dict:
        """Sonify any canonical projection with an explicit acoustic extraction.

        Structured objects are never silently flattened: row-major scans,
        vector norms, graph degree scans, boundary lengths, tensor slices and
        other reductions are recorded in transformation provenance.
        """
        return kernel.sonify_projection(projection_id, mode, title,
                                         options).model_dump(mode="json")

    @mcp.tool
    def math_visualize(view: str = "auto", renderer: str = "auto",
                       title: str | None = None, matrix_id: str | None = None,
                       step_id: str | None = None, system_id: str | None = None,
                       object_id: str | None = None,
                       data: dict | None = None) -> dict:
        """Build a visualization document from kernel objects or inline data.

        Sources: matrix_id (heatmap), step_id (derivation DAG), system_id
        (observation series), object_id (typed object from math_object_create —
        a registered domain adapter chooses the canonical projection, e.g. a
        Distribution becomes a sampled PDF plot with declared sampling loss),
        or inline data: {"xs": [...], "ys": [...]},
        {"points": [[x,y], ...] or [[x,y,z], ...]}, {"matrix": [[...]]},
        {"grid": [[...]]} (3D surface), {"trajectory": [[x,y,z], ...]},
        {"enclosures": [{"x":0,"lower":"...","upper":"..."}]}.

        Full composition: data={"title": str, "layout": {"cols": int},
        "blocks": [{"kind", "title"?, "span"?, "config"?, "bindings"?,
        "data"?}]} assembles a dashboard from generic building blocks.
        Block kinds: point_cloud_3d, trajectory_3d, surface_3d,
        vector_field_3d, plot2d, histogram, heatmap, dag, metric_grid,
        data_table, text, select.  A select block publishes a parameter;
        other blocks bind to it with bindings={"*": "<param>"} (merge option
        object into config) or bindings={"config.key": "<param>"}, giving
        linked interactive panels.  Returns a viz_id plus metadata — never
        the payload.  view: auto|plot2d|dag|heatmap|point_cloud_3d|
        trajectory_3d|surface_3d; renderer: auto|svg|threejs.
        """
        return kernel.viz_create(view, renderer, title, matrix_id, step_id,
                                 system_id, object_id, data).model_dump(mode="json")

    @mcp.tool
    def math_visualize_dag(step_id: str, title: str | None = None) -> dict:
        """Provenance/obligation DAG visualization for any derivation step."""
        return kernel.viz_dag(step_id, title).model_dump(mode="json")

    @mcp.tool
    def math_render_koopman(system_id: str, basis: dict, exact: bool = True,
                            title: str | None = None) -> dict:
        """Koopman mode-visibility visualization for a finite system.

        MathKernel computes the visibility spectrum (exact rational or the
        GPU numeric path); the artifact layer only packages the result.
        """
        return kernel.viz_koopman(system_id, basis, exact, title).model_dump(mode="json")

    @mcp.tool
    def math_export_artifact(viz_id: str, path: str, mode: str = "portable",
                             include_provenance: bool = True,
                             include_reproducibility: bool = True,
                             deterministic: bool = True) -> dict:
        """Export a visualization as a self-contained interactive HTML artifact.

        mode="portable" (default) embeds data, provenance, trust, integrity
        hashes and the full viewer runtime in one .html file that needs no
        server, network, or MathKernel installation.  Returns the path, size
        and SHA-256 — the artifact itself is written to disk, not returned.
        """
        return kernel.viz_export(viz_id, path, mode, include_provenance,
                                 include_reproducibility,
                                 deterministic).model_dump(mode="json")

    @mcp.tool
    def math_sonify(data: list[float], mode: str = "auto", title: str | None = None,
                    options: dict | None = None) -> dict:
        """Create a scientific sonification. mode: auto|scan|harmonic|fourier.

        Mappings from mathematical values to pitch/gain/phase/time are retained
        explicitly as provenance. Hearing a pattern is candidate generation, not proof.
        """
        return kernel.sonify_create(data, mode, title, options).model_dump(mode="json")

    @mcp.tool
    def math_sonify_compare(prediction: list[float], observation: list[float],
                            mode: str = "stereo", title: str | None = None,
                            options: dict | None = None) -> dict:
        """Sonify prediction vs observation. mode=stereo or residual."""
        return kernel.sonify_compare(prediction, observation, mode, title, options).model_dump(mode="json")

    @mcp.tool
    def math_sonification_describe(sonification_id: str) -> dict:
        """Return the complete SonificationDocument including source lineage and mappings."""
        return kernel.sonify_describe(sonification_id).model_dump(mode="json")

    @mcp.tool
    def math_export_audio(sonification_id: str, path: str) -> dict:
        """Render a stored sonification to deterministic lossless WAV and return metadata."""
        return kernel.sonify_export(sonification_id, path).model_dump(mode="json")

    @mcp.tool
    def math_research_artifact_create(title: str = "MathKernel research artifact",
                                      viz_ids: list[str] | None = None,
                                      sonification_ids: list[str] | None = None,
                                      result: dict | None = None,
                                      synchronize: bool = True) -> dict:
        """Assemble a unified multimodal research artifact from stored
        visualization documents (viz_ids from math_visualize) and sonification
        documents (sonification_ids from math_sonify).

        Source lineage, evidence, transformations and trust are merged (trust
        is the weakest evidence in the artifact); where a sonification event
        and a visual block resolve to the same mathematical source, a
        cross-modal synchronization link is derived automatically
        (synchronize=true). Returns artifact_id + summary — never the payload.
        """
        return kernel.research_artifact_create(title, viz_ids, sonification_ids,
                                               result, synchronize).model_dump(mode="json")

    @mcp.tool
    def math_export_research_artifact(artifact_id: str, path: str) -> dict:
        """Export a stored research artifact as one self-contained interactive
        HTML file: embedded visualizations (full block viewer incl. 3D),
        sonification players (WebAudio), cross-modal sync (audio time
        highlights linked visual blocks; selecting a block seeks the audio),
        unified inspectors (Result/Evidence/Provenance/Data/Reproduction/
        Visual Mapping/Audio Mapping/Sync/Annotations), SHA-256 integrity.
        Portable: no server, network, Python or CDN; works from file://.
        Returns path/bytes/SHA-256 metadata, never the file content.
        """
        return kernel.research_artifact_export(artifact_id, path).model_dump(mode="json")

    @mcp.tool
    def math_closure_search(kind: str, params: dict) -> dict:
        """Search short exact closure relations selected by the dynamics.

        kind='cyclic': tuples (k_0..k_{d-1}) with sum_j k_j*mult_j == 0 (mod m)
        and sum |k_j| <= weight_bound — params {m, weight_bound, multipliers:
        [...]} or {m, weight_bound, a_k, d} for equal spacing. kind='binary':
        mask tuples with xor_j (L^{jK})^T w_j = 0, Hamming weight <=
        max_weight — params {rows: [hex], width, d, k_step, max_weight}.
        kind='cyclic_order'/'binary_order': smallest order with an
        irreducible relation (params plus optional d_min/d_max). Add
        irreducible_only=true to filter reducible unions of lower-order
        closures. Meet-in-the-middle; enumeration budgets are capped.
        """
        return kernel.closure_search(kind, **params).model_dump(mode="json")

    @mcp.resource("mathkernel://expression/{expr_id}")
    def expression_resource(expr_id: str) -> str:
        result = kernel.get_expression(expr_id)
        return result.model_dump_json(indent=2)

    @mcp.resource("mathkernel://artifact/{artifact_id}")
    def artifact_resource(artifact_id: str) -> str:
        record = kernel.artifacts.get(artifact_id)
        if record is None:
            return json.dumps({"ok": False, "errors": [f"Unknown artifact_id: {artifact_id}"]}, indent=2)
        return json.dumps(record["artifact"], indent=2)

    @mcp.resource("mathkernel://derivation/{step_id}")
    def derivation_resource(step_id: str) -> str:
        return kernel.derivation_trace(step_id).model_dump_json(indent=2)

    @mcp.prompt
    def solve_and_codegen(equation: str, variable: str, language: str = "typescript") -> str:
        return (
            f"Solve the equation '{equation}' for '{variable}' rigorously:\n"
            "1. math_parse the equation.\n"
            "2. math_infer_structure to see required algebraic capabilities and constraints.\n"
            "3. math_reason (or math_plan + math_execute_plan) to solve with verified candidates.\n"
            f"4. math_codegen with target='solve', variable='{variable}', language='{language}'.\n"
            "5. math_verify_code on the artifact (typecheck + symbolic_roundtrip).\n"
            "Report assumptions, side conditions, and trust level explicitly."
        )

    @mcp.prompt
    def prove_identity(left: str, right: str) -> str:
        return (
            f"Determine whether '{left}' is equivalent to '{right}':\n"
            "1. math_prove_equivalence with formal=true.\n"
            "2. If unknown, math_counterexample to search for a refutation.\n"
            "3. Report the trust level and which engines provided evidence."
        )
else:
    mcp = None

def main() -> None:
    """Console entry point with an actionable optional-transport error."""
    if mcp is None:
        raise SystemExit("FastMCP is not installed. Install the transport with: pip install 'mathkernel[mcp]' (or pip install -e '.[mcp]' from source).")
    from mathkernel.lean_bootstrap import ensure_lean_toolchain, resolve_lean_toolchain, skip_install
    if resolve_lean_toolchain() is None and not skip_install():
        ensure_lean_toolchain()
    mcp.run()


if __name__ == "__main__":
    main()
