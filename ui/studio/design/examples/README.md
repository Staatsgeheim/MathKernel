# Inert examples and fixtures

These files are documentation fixtures. They do not execute mathematics, contact a host, authorize spending, or assert that a feature exists.

`empty.mkstudio.json` is a minimal editor document matching the proposed `mk.studio/1` schema.

`exact_input_fixture.mkstudio.json` illustrates two unresolved authoring nodes and preserves the integer `9007199254740993` as text. Operation identifiers beginning `fixture.` are not real MathKernel registrations. The host binding is null, and importing the file must never trigger computation.

`host_capabilities_fixture.json` describes a deliberately limited mock host. Workflow execution is false. A UI that loops over operations anyway would violate the specification.

`run_observation_fixture.json` demonstrates independent execution, verification, artifact, resource, cost, and freshness axes. It is a **synthetic presentation fixture**, not a calculation or verification result. In particular, result readiness coexists with unknown cleanup and exposure.

The editor schema does not validate operation-specific types, context, claims, execution eligibility, permissions, or cost. Those require real host contracts and separate tests.
