# Compute release qualification

The package remains a development release. Linux/Python 3.12 local CPU is the
qualified execution configuration. SSH, native Slurm, Modal, Runpod and Lambda
are implemented experimental profiles with explicit deployment gates. No live
cloud service, GPU, cluster, Windows or macOS result is implied by fixture tests.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the exact checkpoint results.

## Installation and public contracts

Base MathKernel imports remain independent of compute SDKs. `compute` adds SciPy
and RFC 8785; `compute-modal` pins Modal 1.5.5; `compute-runpod` pins boto3 1.43.90;
`compute-runpod-worker` additionally pins runpod 1.12.0. Lambda uses the `compute`
extra, standard-library HTTPS, and the existing system OpenSSH client. It has no
extra cloud SDK or new executable command family outside `mathkernel-compute`.

`mathkernel-compute schemas` exports 19 JSON Schemas directly from runtime
contracts, including gateway, managed profiles, owned-VM plans/grants/watchdog and
replay. This operation creates no journal, worker or provider connection. Runtime
checks additionally enforce byte/depth limits, canonical binding, policy, ownership
and authorization; passing JSON Schema alone grants nothing.

`examples/compute/exact-request.json` and `numeric-request.json` are complete
bounded requests for the two registered workloads. Follow [README.md](README.md)
for the normal terminal approval/submission flow. Larger domain coverage, arbitrary
object codecs, caches, checkpoints and remote engine integration remain separate
extensions, not implied by installing an adapter.

## Replay

```sh
mathkernel-compute --state-dir compute-state replay JOB_ID > replay.json
mathkernel-compute inspect-replay replay.json
```

New exports use `mk.compute-replay/2`; inspection also accepts the prior version 1.
The manifest binds the original plan, exact bundle, runtime and attempt and lists
external candidate/result hashes. Artifact bytes are not embedded. Preserve those
bytes separately under your data-export policy. A hash does not prove possession,
correctness or trust. Reported historical status remains untrusted input.

Inspection is bounded and offline, rejects duplicate/unknown fields and inconsistent
bindings, and does not import objects, issue a grant, stage files, connect to a
provider or repeat a paid job. A deliberate rerun requires a fresh plan and current
host authority. Neither credentials nor grants are exported. As with any export,
the approved mathematical inputs and operator-written descriptions are data you
choose to disclose; do not put secrets in those fields.

## Reproducible checks

From the source tree:

```sh
python -m pip install '.[compute,compute-modal,compute-runpod,dev,perf]' 'paramiko>=3,<5' build
python -m pytest tests/compute -q
python -m pytest tests/test_jobs.py tests/test_cuboid.py tests/test_engineering_signal.py tests/test_evidence_composition_hardening.py -q
python -m unittest discover -s tests -p test_workflow_runtime.py -v
python -m build
```

Use a fresh wheelhouse for clean resolution, including build dependencies:

```sh
python -m pip download --dest wheelhouse '.[compute,compute-modal,compute-runpod,compute-runpod-worker]' 'setuptools>=77' wheel
python tools/qualify_compute_release.py --wheel dist/mathkernel-1.4.0.dev4-py3-none-any.whl --sdist dist/mathkernel-1.4.0.dev4.tar.gz --wheelhouse wheelhouse --output qualification
```

The qualifier inspects wheel/sdist contents, creates independent virtual
environments for base and each advertised provider extra, rebuilds/installs the
sdist through isolated packaging, runs `pip check`, checks base mathematics and
installed schemas/entrypoints, and saves exact dependency freezes. SDK versions
are checked without making provider calls. Distribution inspection rejects
bytecode, private-key files, journals, job spools, symlinks and runtime artifacts.

The installed local measurement runs the exact and numerical examples, closes the
client after submit, reconnects to the original attempt, retains output, independently
verifies it and retrieves a result. The explicit `--authorize-local` flag on
`examples/compute/measure_local.py` is a host authorization for those local requests.
It reports actual planning, startup, handler, retrieval, verification and cleanup
timings. Queue/provisioning/network stages are marked inapplicable for a local run.
Unattributed polling/serialization overhead remains in the end-to-end measurement;
phase numbers are not falsely summed into a complete model. There is no invented
GPU speedup, price, cold-image timing, calibrated percentile or remote comparison.

## Deployment gates

Before describing any remote profile as stable, qualify the exact account,
OS/Python/SDK/image/runtime and operation combination for clean install, explicit
export/paid authority, lost acknowledgement, controller exit, cancellation,
durable output after provider expiry, local verification, and complete resource
and billing reconciliation. Lambda additionally needs independent watchdog failure
tests and authenticated host-key onboarding. Keep unsupported capabilities explicit.
The planned 114-case design catalog is not a claim that all its cases were run.
