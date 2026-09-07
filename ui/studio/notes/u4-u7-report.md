# Studio 0.3 advanced integration preview

Grounded on live `Staatsgeheim/MathKernel`, branch `mathkernel-studio`, commit
`07f3acd1829addb6c5293f4362f2fdb451b0fbbd` on 7 September 2026. This extends that
branch directly. No default-branch merge or remote publication is included.

## Completion boundary

**This is not a completed U4–U7 release.** The design requires an external,
authoritative workflow host for U4 and a real compute-policy integration for U6.
Neither exists in the supplied repository. Design sections 28.1 and 29 explicitly
exclude adding a scheduler under the UI package. The default host therefore
remains read-only; fixture plans never execute mathematics or spend money.

| Milestone | Delivered in this continuation | Remaining gate |
| --- | --- | --- |
| U4 | Full-digest checks before approval and submission; approximate host clock and expiring controls; request/plan recovery journal; cross-tab Web Locks; explicit reconciliation after remount; consistent snapshot/event/attempt facts; paged run-list refresh; run-to-result identity checks | Real frozen-revision execution, durable host ownership/reconnect, event cursors/streams and operation composition contracts |
| U5 | Existing groups/fragments/history retained; ID-based edge comparison, output-set comparison, explicit versioned subworkflow boundary table and previous-boundary review; changed digest at fixed revision refused | Editable/nested executable subworkflows require a host contract; unsupported constructs remain unavailable |
| U6 | Structured resource/cost/export/alternative review; unknown values preserved; hard-cap limitations explicit; bounded isolated sine-event audition, playback/pause/stop/seek/volume/mute and event table; lifecycle disposal and failure fallback | Real compute authorization/policy integration; general PCM fidelity, mappings, phase/timbre, linked source selection and selection-to-input remain unsupported |
| U7 | 0.3 versioned package, new regression and mounted-DOM suites, unique dialog IDs, source-table column headers, updated docs/skills, strict compiled asset manifest and wheel | Fresh real-browser matrix, Windows/macOS screen readers, localization, reference-hardware performance and long-duration viewer/event memory qualification |

## Behavioral details

- Mutations remain explicit. No mount, import, hover, polling or recovery path
  submits, approves, retries, provisions or cancels a computation.
- Pending journals contain request and plan identifiers only. Legacy 0.2 request
  identifiers remain readable. Corruption, unavailable storage, unretained writes,
  existing pending commands and missing Web Locks prevent new submissions.
- Acknowledgements must match request and plan; accepted outcomes need a run
  reference. The synthetic host also rejects request ID reuse with changed
  payloads or host/workspace/session scope and deep-copies frozen validation data.
- A newer run revision cannot rewrite facts at an unchanged attempt revision or
  mutate an existing event ID. Conflicting duplicates are refused, exact duplicates
  coalesce, and older snapshots cannot overwrite newer observations. Observation
  is manual, bounded snapshots, not an implemented durable event stream.
- Run-result opening requires matching run, document, draft revision and node.
  Existing evidence rendering and host admission rules still apply independently.
- Expiry display is based on host envelope time plus elapsed monotonic client
  time. It is approximate; host enforcement remains authoritative. Held key
  repeats cannot confirm; intentional keyboard activation remains available.
- Audio audition projects only existing resolved sine events. It runs no mappings,
  transforms, normalization, FFTs or mathematical operations. It is explicitly not
  the PCM export: gain is divided by event count and volume starts at 10%.
  Supported bounds: 500 events, 60 seconds, 20–20,000 Hz, gain 0–1. The actual
  device Nyquist bound is checked. Text source alternatives retain exact input
  representation outside the numeric presentation projection.
- The viewer stays in an opaque-origin script-only sandbox with an exact script
  CSP hash, no network permission, and a narrow MessageChannel. Audio objects are
  created only after Play and released on pause, stop, errors and page disposal.

## Verification

| Check | Evidence |
| --- | --- |
| Frontend | 98 passing Vitest tests across 10 files, including mounted jsdom workflows and mocked WebAudio lifecycle |
| Host | 37 unittest cases, including four existing live MathKernel integration cases; no skipped cases |
| Production | Strict TypeScript, Vite shell/workers/viewer builds, asset hash manifest, dependency inventory and licenses |
| Distribution | Optional 0.3.0a1 wheel; manifest validation and isolated extracted-wheel host smoke test |
| Browser | Not executed for this revision: agent-browser installation hit a certificate failure; Playwright browser download subsequently timed out / was blocked by network approval |
| Screen reader / performance | Not executed; no supported-platform or latency claim added |

The previous 0.2 browser report is historical evidence only, not a pass for this
revision. jsdom and mocked AudioContext tests do not validate an actual browser,
audio device, assistive technology, network sandbox, or user-owned compute host.
The original 144-case design catalog stays unchanged and is not an executed pass
report. Tests relevant to this continuation live in `advanced.test.ts`,
`workflow-ui.test.tsx`, `audio-ui.test.ts`, `host.test.ts`, and host
`test_workflow.py`, alongside the existing suites.

## Reproduce

From the repository root, with Node 22.12+ and Python 3.11+:

```bash
npm --prefix ui/studio ci
npm --prefix ui/studio test
npm --prefix ui/studio run build
PYTHONPATH=src:ui/studio/host/src python -m unittest discover -s ui/studio/host/tests -v
python -m pip wheel --no-deps --no-build-isolation ./ui/studio/host -w dist
```

The handoff includes source with built assets, the optional wheel, the Git branch
bundle, a base-to-result patch and SHA-256 checksums. Install the core from source
and the optional wheel to run Studio without Node. On Windows, set `PYTHONPATH`
with the platform's path separator if running source tests directly.
