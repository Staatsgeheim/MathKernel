# OpenAI Navier–Stokes adversarial audit case

**Result of this run: no mathematical disproof found. Full formal verification
was not executed.** This example records a targeted first audit and supplies
repeatable local checks; it is not a complete dependency audit or a review of all
166 pages of the proof.

`case.json` pins the OpenAI repository and the independently authored Formal
Conjectures statement. `spec.json` names the actual C/D target declarations and
requires the project's Lean 4.34.0-rc2 toolchain and standard axiom whitelist.
The core MathKernel arithmetic-certificate toolchain stays at 4.33.0; external
projects have their own pinned environments and do not mutate that installation.

## Findings from source inspection

The final target statements existentially quantify initial data and smooth force,
then deny existence of a global solution in the declared class. The inspected
solution definitions require the PDE, incompressibility, initial condition and
space-time smoothness. The whole-space class also carries square integrability
and a uniform energy bound; the periodic class includes pressure periodicity.
The corresponding definition bodies were inspected against the pinned independent
reference. No obvious mismatch was identified; this was a manual targeted review,
not kernel-checked definitional equivalence.

The `R3ActualCandidate` adapter obtains a witness from
`ActualCandidateAssembly.selected_witness`. The assembly also contains
`selected_candidate : ProblemStatement.candidateStatement`. Consequently, an
older file labelling the bare proposition OPEN is **not evidence** that the final
artifact lacks a proof. Likewise, intentional `sorry` challenge placeholders are
not a demonstrated dependency of the submitted targets. No transitive axiom audit
was executed, so the repository's zero-`sorry` metadata remains self-reported.

The concrete trust-workflow issue is that ordinary submission builds must not
precede a trusted Comparator boundary. Comparator explicitly assumes trusted
challenge/import/build files and a checking environment not compromised by prior
submission builds. Its documented sandbox restrictions also matter. This is a
verification precondition, **not a flaw proved to exist in the Navier–Stokes
mathematics**, and no exploit against the submitted artifact was demonstrated.

## Executed local mathematics

```sh
PYTHONPATH=src python examples/audits/openai_navier_stokes/local_checks.py --output /outside/repository/local-checks.json
```

The script uses real MathKernel calls, with exact rational expressions. It checks
13 identities covering anisotropic support volume, kinetic-energy and L3 scaling,
radial-gradient/enstrophy scaling, Reynolds and advection rates, relative diffusion,
the reduced chart Jacobian, and viscosity rescaling. Three outward interval checks
establish positive margins on the closed superset `0 <= h <= 1/100`.

It also adds 1 to each right-hand side and evaluates the resulting deliberately
false claim at rational `h=1/200`, `eta=1/2`, `s=2` where applicable. These 13
negative controls must be exactly refuted. They are test mutations, **not** errors
attributed to the paper. Output includes full claim-specific MathKernel evidence.

The scaling assumptions are reviewed inputs. Correct exponent arithmetic does not
prove a field exists with those exponents, that every momentum-residual derivative
extends smoothly, or that a nonlinear induction/convergence estimate is valid.
No numerical Navier–Stokes simulation is used as a surrogate for these obligations.

## Remaining decisive obligations

A separately provisioned trusted reference, compatible toolchain, trusted
dependency caches and pinned Comparator/Landrun/lean4export/nanoda binaries are
needed for actual replay. Use an unprivileged disposable Linux account and follow
[the replay guide](../../../skills/mathkernel/formal-project-audit.md). Do not run
`lake build` on an untrusted submission before that setup.

Beyond replay, an independent mathematical audit must establish paper-to-Lean
statement alignment and review the infinite-order forcing extension, nonlinear
correction estimates, convergence and final uniqueness/periodization bridges.
This package does not certify those obligations as solved.
