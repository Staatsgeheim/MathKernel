import { useState } from 'react';
import { Dialog, JsonView, Text } from '../app/components';
import { deleteRecovery, restoreRecovery, type RecoveryEntry } from './recovery';
import type { StudioDocument } from './document';
import { downloadJson } from './CompositionPanel';
import { compareDocuments, type Difference } from './composition';

export function RecoveryBrowser({
  current,
  entries,
  onRestore,
  onClose,
}: {
  current: StudioDocument;
  entries: RecoveryEntry[];
  onRestore: (d: StudioDocument) => void;
  onClose: () => void;
}) {
  const [copies, setCopies] = useState(entries),
    [remove, setRemove] = useState<RecoveryEntry | null>(null),
    [error, setError] = useState('');
  const [comparison, setComparison] = useState<Difference[] | null>(null);
  return (
    <Dialog title="Saved drafts in this browser" onClose={onClose}>
      <p>
        Copies belong to the displayed host and workspace. Restoring does not connect or execute.
        Export the current draft before replacing it.
      </p>
      {!copies.length && <p>No saved drafts in this scope.</p>}
      {copies.map((e) => {
        let title: string;
        try {
          title = restoreRecovery(e).identity.title;
        } catch {
          title = 'Invalid recovery document';
        }
        return (
          <article className="group-card" key={e.key}>
            <h3>
              <Text>{title}</Text>
            </h3>
            <p>
              Saved {e.updatedAt} · storage revision {e.version}
            </p>
            <div className="toolbar">
              <button
                onClick={() => {
                  try {
                    onRestore(restoreRecovery(e));
                  } catch (error) {
                    setError(String(error));
                  }
                }}
              >
                Restore this draft
              </button>
              <button
                onClick={() => {
                  try {
                    downloadJson(restoreRecovery(e), 'recovered-draft.mkstudio.json');
                  } catch (error) {
                    setError(String(error));
                  }
                }}
              >
                Export copy
              </button>
              <button
                onClick={() => {
                  try {
                    setComparison(compareDocuments(restoreRecovery(e), current));
                  } catch (error) {
                    setError(String(error));
                  }
                }}
              >
                Compare with current draft
              </button>
              <button onClick={() => setRemove(e)}>Delete saved copy…</button>
            </div>
          </article>
        );
      })}
      {comparison && (
        <section>
          <h3>Saved copy → current draft</h3>
          <p>
            {comparison.length} changed fields; first 100 displayed. Comparison does not alter
            either document.
          </p>
          <JsonView value={comparison.slice(0, 100)} />
          <button
            onClick={() =>
              downloadJson(
                { schema: 'mk.studio.diff/1', changes: comparison },
                'recovery-comparison.json',
              )
            }
          >
            Export complete comparison
          </button>
        </section>
      )}
      {remove && (
        <section className="warning">
          <p>
            Delete this saved browser copy? The current draft and host resources are unaffected.
          </p>
          <button
            onClick={async () => {
              try {
                await deleteRecovery(remove.key, remove.version);
                setCopies((old) => old.filter((e) => e.key !== remove.key));
                setRemove(null);
              } catch (e) {
                setError(String(e));
              }
            }}
          >
            Confirm deletion
          </button>
          <button onClick={() => setRemove(null)}>Keep copy</button>
        </section>
      )}
      {error && (
        <p className="warning" role="alert">
          {error}
        </p>
      )}
    </Dialog>
  );
}
