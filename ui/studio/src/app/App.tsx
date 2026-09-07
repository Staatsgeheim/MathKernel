import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '../editor/Canvas';
import { ConnectDialog } from '../editor/ConnectDialog';
import {
  diagnostics,
  descriptor,
  execute,
  history,
  redo,
  undo,
  type Command,
} from '../editor/commands';
import {
  emptyDocument,
  makeInput,
  newId,
  serializeDocument,
  type DraftEdge,
  type StudioDocument,
} from '../editor/document';
import { readDocument } from '../editor/import';
import {
  deleteRecovery,
  listRecovery,
  loadRecovery,
  recoveryScope,
  restoreRecovery,
  saveRecovery,
  type RecoveryEntry,
} from '../editor/recovery';
import { NodeInspector } from '../inspectors/NodeInspector';
import { HostCommandService, HostError } from '../host/client';
import type { Handshake, Operation, ResultObservation } from '../host/contracts';
import { ResultInspector } from '../evidence/ResultInspector';
import { Dialog, Field, JsonView, Text, ViewBoundary } from './components';
import { CONTROL_BYTES, parseJson } from '../security/json';

const client = new HostCommandService();
const NO_RUN =
  'Workflow execution is unavailable in this Studio build. The host must supply validation, planning and execution contracts.';
export function App() {
  const [state, setState] = useState(() => history(emptyDocument()));
  const [selected, setSelected] = useState<string[]>([]);
  const [host, setHost] = useState<Handshake | null>(null);
  const [catalog, setCatalog] = useState<Operation[]>([]);
  const [query, setQuery] = useState('');
  const [view, setView] = useState<'canvas' | 'outline'>(() =>
    matchMedia('(max-width: 1000px)').matches ? 'outline' : 'canvas',
  );
  const [mode, setMode] = useState<'author' | 'inspect'>('author');
  const [message, setMessage] = useState(() =>
    location.pathname === '/studio/' || location.pathname === '/studio'
      ? 'Authoring offline. No computation has been requested.'
      : 'This deep link cannot be resolved by this preview. No document or run was loaded. Use import or an explicit host result reference.',
  );
  const [connection, setConnection] = useState(false);
  const [code, setCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [connectionState, setConnectionState] = useState('Disconnected');
  const [connecting, setConnecting] = useState<{ edge?: DraftEdge; oldId?: string } | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [importPreview, setImportPreview] = useState<StudioDocument | null>(null);
  const [importing, setImporting] = useState(false);
  const importAbort = useRef<AbortController | null>(null);
  const [observations, setObservations] = useState<{ result_ref: string; source_ref: string }[]>(
    [],
  );
  const [resultOffset, setResultOffset] = useState<number | null>(0);
  const [observation, setObservation] = useState<ResultObservation | null>(null);
  const observationEpoch = useRef(0);
  const [recoveryEnabled, setRecoveryEnabled] = useState(false);
  const [recovery, setRecovery] = useState<RecoveryEntry | null>(null);
  const [recoveryStatus, setRecoveryStatus] = useState(
    'Recovery is off; export a file to keep your work.',
  );
  const recoveryVersion = useRef(0);
  const recoveryQueue = useRef(Promise.resolve());
  const [help, setHelp] = useState(false);
  const [pendingCheck, setPendingCheck] = useState(false);
  const stateRef = useRef(state);
  const commit = useCallback((next: typeof state) => {
    stateRef.current = next;
    setState(next);
  }, []);
  const doc = state.document;
  const hostId = host?.host_instance_id ?? 'offline';
  const workspaceId = host?.workspace_id ?? 'local';
  const scope = recoveryScope(hostId, workspaceId, doc.identity.document_id);
  const activeScope = useRef(scope);
  activeScope.current = scope;
  const activeNode = doc.authoring.nodes.find((n) => n.id === selected[0]);
  const problems = useMemo(() => diagnostics(doc, catalog), [doc, catalog]);
  const matches = useMemo(() => {
    const term = query.toLocaleLowerCase();
    return catalog.filter((o) =>
      [o.title, o.operation_ref, o.domain, o.description, ...o.input_types, ...o.output_types]
        .join(' ')
        .toLocaleLowerCase()
        .includes(term),
    );
  }, [catalog, query]);
  const dispatch = useCallback(
    (command: Command): boolean => {
      try {
        const next = execute(stateRef.current, command, catalog);
        commit(next);
        return true;
      } catch (error) {
        setMessage(error instanceof Error ? error.message.slice(0, 1000) : 'Edit rejected.');
        return false;
      }
    },
    [commit, catalog],
  );
  const clearHost = useCallback(() => {
    observationEpoch.current++;
    client.disconnect();
    setHost(null);
    setCatalog([]);
    setObservations([]);
    setObservation(null);
    setResultOffset(0);
    setConnectionState('Disconnected');
    setRecoveryEnabled(false);
  }, []);
  function hostError(error: unknown) {
    if (error instanceof HostError && [401, 403].includes(error.status)) {
      clearHost();
      setMessage(
        'Session or permission lost. Private observations were cleared; draft intent is preserved.',
      );
    } else {
      setConnectionState('Stale / read failed');
      setMessage(error instanceof Error ? error.message.slice(0, 1000) : 'Host read failed.');
    }
  }
  async function connectHost() {
    if (busy) return;
    setBusy(true);
    clearHost();
    setConnectionState('Connecting');
    try {
      const h = await client.connect(code || undefined);
      setCode('');
      const entries = h.features.catalog_read ? await client.catalog() : [];
      setHost(h);
      setCatalog(entries);
      setConnectionState(h.test_host ? 'TEST HOST' : 'Connected');
      setConnection(false);
      setMessage(`${entries.length} catalog entries loaded. ${NO_RUN}`);
    } catch (e) {
      setCode('');
      clearHost();
      hostError(e);
    } finally {
      setBusy(false);
    }
  }
  function addInput(kind: 'integer' | 'rational' | 'real') {
    const n = makeInput(kind);
    if (
      dispatch({
        type: 'add',
        node: n,
        position: {
          x: 60 + (doc.authoring.nodes.length % 3) * 310,
          y: 60 + Math.floor(doc.authoring.nodes.length / 3) * 210,
        },
      })
    )
      setSelected([n.id]);
  }
  function addOperation(op: Operation) {
    const n = {
      id: newId(),
      kind: 'operation' as const,
      label: op.title,
      operation_ref: op.operation_ref,
      operation_version: op.operation_version,
      schema_digest: op.schema_digest,
      parameter_drafts: {},
      input_bindings: {},
    };
    if (
      dispatch({
        type: 'add',
        node: n,
        position: {
          x: 60 + (doc.authoring.nodes.length % 3) * 310,
          y: 60 + Math.floor(doc.authoring.nodes.length / 3) * 210,
        },
      })
    )
      setSelected([n.id]);
  }
  function exportDocument() {
    try {
      const blob = new Blob([serializeDocument(doc)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = `${doc.identity.title.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 80) || 'experiment'}.mkstudio.json`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setMessage(
        'Studio document exported. This is authoring data, not an executable workflow or certificate.',
      );
    } catch (e) {
      setMessage(String(e));
    }
  }
  async function importFile(file: File) {
    importAbort.current?.abort();
    const abort = new AbortController();
    importAbort.current = abort;
    setImporting(true);
    try {
      setImportPreview(await readDocument(file, abort.signal));
    } catch (e) {
      setMessage(String(e));
    } finally {
      setImporting(false);
    }
  }
  async function importCandidate(file: File) {
    if (file.size > CONTROL_BYTES) {
      setMessage('Candidate file exceeds the 2 MiB inspection budget.');
      return;
    }
    const epoch = observationEpoch.current;
    try {
      const result = parseJson(await file.text(), CONTROL_BYTES);
      if (epoch !== observationEpoch.current) return;
      setObservation({
        binding: {
          result_ref: newId(),
          source_ref: 'imported-file',
          source_revision: 'untrusted',
          run_ref: null,
          document_id: null,
          draft_revision: null,
          node_id: null,
        },
        result,
        claim_trust: {},
        admission: 'candidate',
        observed_at: new Date().toISOString(),
        scope: { host_instance_id: 'untrusted-file', workspace_id: 'unresolved' },
      });
      setMode('inspect');
    } catch (e) {
      setMessage(String(e));
    }
  }
  async function enableRecovery() {
    try {
      const current =
        (await loadRecovery(scope)) ?? (await listRecovery(hostId, workspaceId))[0] ?? null;
      if (activeScope.current !== scope) return;
      recoveryVersion.current = current?.version ?? 0;
      if (current) {
        setRecovery(current);
        setRecoveryStatus('Existing recovery copy found; choose restore or discard before saving.');
      } else setRecoveryEnabled(true);
    } catch (e) {
      setRecoveryStatus(`Recovery unavailable: ${String(e)} Export a file.`);
    }
  }
  useEffect(() => {
    setRecoveryEnabled(false);
    setRecovery(null);
    recoveryVersion.current = 0;
    setRecoveryStatus('Recovery is off; export a file to keep your work.');
  }, [scope]);
  useEffect(() => {
    if (!recoveryEnabled || recovery) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      recoveryQueue.current = recoveryQueue.current.then(async () => {
        if (cancelled) return;
        try {
          const revision = await saveRecovery(scope, doc, recoveryVersion.current);
          if (activeScope.current !== scope) return;
          recoveryVersion.current = revision;
          if (!cancelled)
            setRecoveryStatus(
              `Saved in this browser · draft ${doc.authoring.revision}, layout ${doc.presentation.revision}. Not a backup.`,
            );
        } catch (e) {
          if (!cancelled) {
            setRecoveryEnabled(false);
            setRecoveryStatus(String(e));
          }
        }
      });
    }, 400);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [doc, scope, recoveryEnabled, recovery]);
  useEffect(() => {
    const listener = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (
        e.isComposing ||
        target.closest('input, textarea, select, [contenteditable="true"], dialog')
      )
        return;
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        commit(e.shiftKey ? redo(stateRef.current) : undo(stateRef.current));
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        commit(redo(stateRef.current));
      }
      if (e.key === 'Delete' && selected.length) {
        e.preventDefault();
        setDeleting(true);
      }
    };
    window.addEventListener('keydown', listener);
    return () => window.removeEventListener('keydown', listener);
  }, [selected, commit]);
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (doc.authoring.nodes.length) e.preventDefault();
    };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [doc.authoring.nodes.length]);
  const select = useCallback((ids: string[]) => setSelected(ids), []);
  const canvasConnect = useCallback(
    (edge: DraftEdge, oldId?: string) => setConnecting({ edge, oldId }),
    [],
  );
  const affected = doc.authoring.edges.filter(
    (e) => selected.includes(e.source_node) || selected.includes(e.target_node),
  ).length;
  return (
    <div className="studio-app">
      <a className="skip-link" href="#workbench">
        Skip to workspace
      </a>
      <header className="app-bar">
        <div className="brand">
          <span className="brand-symbol" aria-hidden="true">
            Mκ
          </span>
          <div>
            <strong>
              MathKernel <span>Studio</span>
            </strong>
            <small>Catalog · authoring · inspection preview</small>
          </div>
        </div>
        <div className="header-actions">
          <button onClick={() => setConnection(true)}>{connectionState}</button>
          <button onClick={() => setHelp(true)}>Help & capabilities</button>
        </div>
      </header>
      {host?.test_host && (
        <div className="test-banner">
          TEST HOST — synthetic contracts and observations. No mathematics is executed.
        </div>
      )}
      <div className="document-bar">
        <div className="document-title">
          <Field
            label="Document title"
            value={doc.identity.title}
            onCommit={(title) => dispatch({ type: 'title', title })}
          />
          <span className="badge">
            Draft {doc.authoring.revision} · layout {doc.presentation.revision}
          </span>
        </div>
        <div className="toolbar">
          <button
            onClick={() => {
              setMode('author');
              setPendingCheck(true);
              setMessage(
                `Editor integrity checked at draft ${doc.authoring.revision}. ${problems.length} advisory/structural notices. No host validation occurred.`,
              );
            }}
          >
            Check draft
          </button>
          <button disabled title={NO_RUN}>
            Validate with host
          </button>
          <button disabled title={NO_RUN}>
            Plan
          </button>
          <button disabled title={NO_RUN}>
            Run workflow
          </button>
        </div>
      </div>
      <div className="workspace-grid">
        <aside className="catalog-panel" aria-label="Operation palette">
          <header className="panel-header">
            <h2>Operations</h2>
            <span className="muted">{catalog.length}</span>
          </header>
          <label className="search field">
            Search catalog
            <input
              type="search"
              placeholder="Name, type or domain…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
          <section className="input-palette">
            <p className="eyebrow">Local input drafts</p>
            <div className="input-buttons">
              <button onClick={() => addInput('integer')}>ℤ Integer</button>
              <button onClick={() => addInput('rational')}>ℚ Rational</button>
              <button onClick={() => addInput('real')}>ℝ Numerical</button>
            </div>
          </section>
          <div className="catalog-list">
            {!host && (
              <div className="empty-panel">
                <h3>Connect your local host</h3>
                <p>
                  Browse its real operation catalog. Local inputs and document editing work offline.
                </p>
                <button onClick={() => setConnection(true)}>Connect</button>
              </div>
            )}
            {matches.slice(0, 150).map((op) => (
              <article className="catalog-item" key={op.descriptor_id}>
                <div>
                  <small>
                    <Text>{op.domain}</Text>
                  </small>
                  <strong>
                    <Text>{op.title}</Text>
                  </strong>
                  <p>
                    {op.composition} · {op.availability}
                  </p>
                </div>
                <button aria-label={`Add ${op.title} to draft`} onClick={() => addOperation(op)}>
                  Add
                </button>
              </article>
            ))}
            {matches.length > 150 && (
              <p className="notice">Showing 150 of {matches.length}. Refine your search.</p>
            )}
            {host && matches.length === 0 && <p className="empty-panel">No matching operations.</p>}
          </div>
          <footer className="catalog-footer">
            <button
              onClick={() => {
                setMode('inspect');
                setObservation(null);
              }}
            >
              Inspect results
            </button>
          </footer>
        </aside>
        <main id="workbench" tabIndex={-1} className="workbench">
          <header className="workbench-bar">
            <div className="segmented">
              <button aria-pressed={mode === 'author'} onClick={() => setMode('author')}>
                Author
              </button>
              <button aria-pressed={mode === 'inspect'} onClick={() => setMode('inspect')}>
                Inspect
              </button>
            </div>
            {mode === 'author' && (
              <div className="segmented">
                <button aria-pressed={view === 'canvas'} onClick={() => setView('canvas')}>
                  Canvas
                </button>
                <button aria-pressed={view === 'outline'} onClick={() => setView('outline')}>
                  Outline
                </button>
              </div>
            )}
          </header>
          {mode === 'author' ? (
            <>
              <div className="editor-toolbar">
                <button
                  disabled={!state.past.length}
                  onClick={() => commit(undo(stateRef.current))}
                >
                  Undo
                </button>
                <button
                  disabled={!state.future.length}
                  onClick={() => commit(redo(stateRef.current))}
                >
                  Redo
                </button>
                <button disabled={doc.authoring.nodes.length < 2} onClick={() => setConnecting({})}>
                  Connect
                </button>
                <button
                  disabled={!selected.length}
                  onClick={() => dispatch({ type: 'duplicate', nodeIds: selected })}
                >
                  Duplicate
                </button>
                <button disabled={!selected.length} onClick={() => setDeleting(true)}>
                  Delete
                </button>
                <button
                  disabled={!doc.authoring.nodes.length}
                  onClick={() =>
                    dispatch({
                      type: 'move',
                      positions: Object.fromEntries(
                        doc.authoring.nodes.map((n, i) => [
                          n.id,
                          { x: 60 + (i % 3) * 310, y: 60 + Math.floor(i / 3) * 210 },
                        ]),
                      ),
                    })
                  }
                >
                  Arrange
                </button>
              </div>
              {view === 'canvas' && doc.authoring.nodes.length <= 200 ? (
                <Canvas
                  document={doc}
                  catalog={catalog}
                  selected={selected}
                  onSelect={select}
                  command={dispatch}
                  connect={canvasConnect}
                />
              ) : (
                <div className="outline">
                  <h2>Graph outline</h2>
                  {doc.authoring.nodes.length > 200 && (
                    <p className="notice">
                      Large draft: use the outline. All nodes remain in the saved document.
                    </p>
                  )}
                  {!doc.authoring.nodes.length && (
                    <p>Add an input or a catalog operation to begin.</p>
                  )}
                  {doc.authoring.nodes.map((n) => (
                    <article
                      key={n.id}
                      className={`outline-node ${selected.includes(n.id) ? 'selected' : ''}`}
                    >
                      <label className="check">
                        <input
                          type="checkbox"
                          checked={selected.includes(n.id)}
                          onChange={(e) =>
                            setSelected(
                              e.target.checked
                                ? [...selected, n.id]
                                : selected.filter((v) => v !== n.id),
                            )
                          }
                        />
                        <Text>{n.label}</Text>
                      </label>
                      <code>
                        <Text>{n.operation_ref ?? 'Local input draft'}</Text>
                      </code>
                      <button onClick={() => setSelected([n.id])}>Inspect node</button>
                      <span className="badge">
                        {descriptor(n, catalog)?.composition ?? 'unresolved'} · not run
                      </span>
                    </article>
                  ))}
                  <h3>Connections</h3>
                  {doc.authoring.edges.map((e) => (
                    <div className="outline-edge" key={e.id}>
                      <span>
                        <Text>{`${doc.authoring.nodes.find((n) => n.id === e.source_node)?.label} [${e.source_port}] → ${doc.authoring.nodes.find((n) => n.id === e.target_node)?.label} [${e.target_port}]`}</Text>
                      </span>
                      <button onClick={() => setConnecting({ edge: e, oldId: e.id })}>
                        Reconnect
                      </button>
                      <button onClick={() => dispatch({ type: 'disconnect', edgeId: e.id })}>
                        Disconnect
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="inspection-workspace">
              <div className="toolbar">
                <label className="file-button">
                  Open candidate JSON
                  <input
                    type="file"
                    accept=".json,application/json"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) void importCandidate(f);
                      e.target.value = '';
                    }}
                  />
                </label>
                <button
                  disabled={!host?.features.result_inspect || busy || resultOffset === null}
                  onClick={async () => {
                    if (busy) return;
                    const epoch = observationEpoch.current;
                    setBusy(true);
                    try {
                      const page = await client.results(resultOffset ?? 0);
                      if (epoch === observationEpoch.current) {
                        setObservations((old) => [...old, ...page.results]);
                        setResultOffset(page.next_offset);
                      }
                    } catch (e) {
                      hostError(e);
                    } finally {
                      setBusy(false);
                    }
                  }}
                >
                  Load host result references
                </button>
              </div>
              {!host?.features.result_inspect && (
                <p className="notice">
                  No published host results. Imported JSON is available for untrusted, inert
                  inspection.
                </p>
              )}
              <div className="result-links">
                {observations.map((o) => (
                  <button
                    key={o.result_ref}
                    disabled={busy}
                    onClick={async () => {
                      const epoch = observationEpoch.current;
                      setBusy(true);
                      try {
                        const r = await client.result(o.result_ref);
                        if (epoch === observationEpoch.current) setObservation(r);
                      } catch (e) {
                        hostError(e);
                      } finally {
                        setBusy(false);
                      }
                    }}
                  >
                    <Text>{o.result_ref}</Text>
                  </button>
                ))}
              </div>
              {observation ? (
                <ViewBoundary key={`${observation.binding.result_ref}-${observation.observed_at}`}>
                  <ResultInspector
                    observation={observation}
                    client={client}
                    onHostError={hostError}
                  />
                </ViewBoundary>
              ) : (
                <div className="empty-panel">
                  <h2>Inspect facts, claim by claim</h2>
                  <p>
                    Select an existing host result or open a candidate file. Exact values,
                    assumptions and evidence stay attached to their source.
                  </p>
                </div>
              )}
            </div>
          )}
        </main>
        <aside className="properties-panel" aria-label="Node properties">
          <header className="panel-header">
            <h2>Inspector</h2>
            <span className="muted">
              {selected.length > 1 ? `${selected.length} selected` : 'Draft'}
            </span>
          </header>
          {activeNode && mode === 'author' ? (
            <NodeInspector
              key={activeNode.id}
              node={activeNode}
              position={doc.presentation.node_positions[activeNode.id] ?? { x: 0, y: 0 }}
              catalog={catalog}
              document={doc}
              command={dispatch}
            />
          ) : (
            <div className="empty-panel">
              <h3>{mode === 'author' ? 'Select a node' : 'Read-only result view'}</h3>
              <p>
                {mode === 'author'
                  ? 'Inspect parameters, exact inputs, port metadata and source identity.'
                  : 'The draft remains separate from this observation. Return to Author to continue editing.'}
              </p>
              <p className="notice">{NO_RUN}</p>
            </div>
          )}
        </aside>
      </div>
      <footer className="bottom-panel">
        <details open={pendingCheck || undefined}>
          <summary>
            Problems · {problems.length}{' '}
            <span className="muted">
              Draft {doc.authoring.revision} · editor and advisory checks only
            </span>
          </summary>
          <div className="problem-list">
            {problems.slice(0, 100).map((p, i) => (
              <div key={`${p.code}-${i}`}>
                <span className="badge">{p.layer}</span>
                <span>
                  <Text>{p.message}</Text>
                </span>
                {p.nodeId && (
                  <button
                    onClick={() => {
                      setMode('author');
                      setSelected([p.nodeId!]);
                    }}
                  >
                    Inspect node
                  </button>
                )}
              </div>
            ))}
            {!problems.length && (
              <p>No structural notices. Host mathematical validation has not occurred.</p>
            )}
          </div>
        </details>
        <p className="live-status" role="status" aria-live="polite">
          <Text>{message}</Text>
        </p>
        <div className="save-bar">
          <p className="muted">
            <Text>{recoveryStatus}</Text>
          </p>
          <div className="toolbar">
            <button
              onClick={() => {
                if (recoveryEnabled) {
                  setRecoveryEnabled(false);
                  setRecoveryStatus(
                    'Recovery paused. Existing copy retained; export current edits.',
                  );
                } else void enableRecovery();
              }}
            >
              {recoveryEnabled ? 'Pause recovery' : 'Enable recovery'}
            </button>
            <label className="file-button">
              Import draft
              <input
                type="file"
                accept=".mkstudio.json,application/json"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void importFile(f);
                  e.target.value = '';
                }}
              />
            </label>
            <button className="primary" onClick={exportDocument}>
              Export document
            </button>
          </div>
        </div>
      </footer>
      {connection && (
        <Dialog title="Connect to this local host" onClose={() => setConnection(false)}>
          <p>
            Enter the one-time code printed by the MathKernel Studio process. It expires after 10
            minutes; sessions last one hour. Restart the process for a new code.
          </p>
          <p>Destination: {window.location.origin}. Imported documents cannot change it.</p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void connectHost();
            }}
          >
            <label className="field">
              Connection code (leave blank to reuse this browser session)
              <input
                type="password"
                autoComplete="off"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                maxLength={128}
              />
            </label>
            <div className="dialog-actions">
              <button
                type="button"
                onClick={() => {
                  clearHost();
                  setConnection(false);
                }}
              >
                Disconnect UI
              </button>
              <button className="primary" disabled={busy}>
                {busy ? 'Connecting…' : 'Connect'}
              </button>
            </div>
          </form>
        </Dialog>
      )}
      {connecting && (
        <ConnectDialog
          document={doc}
          catalog={catalog}
          initial={connecting.edge}
          oldId={connecting.oldId}
          onClose={() => setConnecting(null)}
          onApply={(edge, replace, oldId) =>
            dispatch(
              oldId
                ? { type: 'reconnect', edgeId: oldId, edge, replace }
                : { type: 'connect', edge, replace },
            )
          }
        />
      )}
      {deleting && (
        <Dialog title="Delete from this draft?" onClose={() => setDeleting(false)}>
          <p>
            Remove {selected.length} selected node(s), {affected} incident connection(s), and their
            desired-output selections in one undoable edit.
          </p>
          <div className="dialog-actions">
            <button onClick={() => setDeleting(false)}>Cancel</button>
            <button
              onClick={() => {
                if (dispatch({ type: 'delete', nodeIds: selected })) {
                  setSelected([]);
                  setDeleting(false);
                }
              }}
            >
              Delete from draft
            </button>
          </div>
        </Dialog>
      )}
      {importing && (
        <Dialog title="Checking import" onClose={() => importAbort.current?.abort()}>
          <p>Parsing in a bounded worker. Current draft is unchanged.</p>
          <button onClick={() => importAbort.current?.abort()}>Cancel import</button>
        </Dialog>
      )}
      {importPreview && (
        <Dialog title="Review imported authoring document" onClose={() => setImportPreview(null)}>
          <h3>
            <Text>{importPreview.identity.title}</Text>
          </h3>
          <p>
            {importPreview.authoring.nodes.length} nodes · {importPreview.authoring.edges.length}{' '}
            connections. Host and object references remain unresolved. No mathematics or URLs will
            be executed.
          </p>
          <p>
            This replaces the current editor document. Export current work first if you want to keep
            a separate copy.
          </p>
          <JsonView
            value={{
              operations: importPreview.authoring.nodes.map((n) => n.operation_ref),
              host_binding: importPreview.host_binding,
            }}
          />
          <div className="dialog-actions">
            <button onClick={exportDocument}>Export current draft</button>
            <button
              className="primary"
              onClick={() => {
                commit(history(importPreview));
                setSelected([]);
                setImportPreview(null);
                setMode('author');
                setMessage(
                  'Imported as inert authoring intent. Host references were not contacted.',
                );
              }}
            >
              Open imported draft
            </button>
          </div>
        </Dialog>
      )}
      {recovery && (
        <Dialog title="Recovery copy found" onClose={() => setRecovery(null)}>
          <p>
            Saved {recovery.updatedAt}. Restore preserves draft intent only; it will not reconnect
            or execute.
          </p>
          <div className="dialog-actions">
            <button onClick={exportDocument}>Export current draft</button>
            <button
              onClick={async () => {
                try {
                  await deleteRecovery(recovery.key, recovery.version);
                  recoveryVersion.current = 0;
                  setRecovery(null);
                  setRecoveryEnabled(true);
                } catch (e) {
                  setRecoveryStatus(String(e));
                }
              }}
            >
              Discard recovery copy
            </button>
            <button
              className="primary"
              onClick={() => {
                try {
                  commit(history(restoreRecovery(recovery)));
                  setSelected([]);
                  setRecovery(null);
                  setRecoveryEnabled(false);
                  setRecoveryStatus(
                    'Recovery restored. Enable recovery again to resume saving this document.',
                  );
                } catch (e) {
                  setRecoveryStatus(String(e));
                }
              }}
            >
              Restore copy
            </button>
          </div>
        </Dialog>
      )}
      {help && (
        <Dialog title="Studio preview — capabilities and boundaries" onClose={() => setHelp(false)}>
          <p>Build 0.1.0-alpha.1 · document mk.studio/1 · protocol studio-host/1</p>
          <p>
            Add from the palette, select a node, edit in the inspector, and use Connect to wire
            ports without dragging. Outline exposes every connection. Use position fields or Arrange
            to move nodes without dragging. Undo/redo changes drafts only.
          </p>
          <p>
            Exact mathematical integers and rationals use text, including values above 2⁵³.
            Parameter text is saved to the draft as you type; undo groups each field edit. Labels
            and position fields commit on leaving the field. Export preserves invalid parameter
            drafts. Browser recovery is optional, bounded and not a backup.
          </p>
          <p>{NO_RUN}</p>
          <p>
            Only structured text viewers are enabled. Active HTML, plots, audio, remote compute,
            approvals and workflow execution remain unavailable.
          </p>
          <JsonView
            value={host ?? { connection: 'offline', authoring_draft: true }}
            label="Host capability summary"
          />
        </Dialog>
      )}
    </div>
  );
}
