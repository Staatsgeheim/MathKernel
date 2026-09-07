import { randomId } from '../security/identity';
import { Component, useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { displayText } from '../security/json';
export function Text({ children }: { children: string }) {
  return <>{displayText(children)}</>;
}
export function JsonView({ value, label = 'Structured data' }: { value: unknown; label?: string }) {
  const [expanded, setExpanded] = useState(false);
  const text = displayText(JSON.stringify(value, null, 2) ?? String(value));
  return (
    <section className="json-view">
      <pre aria-label={label} tabIndex={0}>
        {expanded ? text.slice(0, 65536) : text.slice(0, 4000)}
      </pre>
      {text.length > 4000 && (
        <button onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Collapse' : 'Show more (up to 64 KiB)'}
        </button>
      )}
      {text.length > (expanded ? 65536 : 4000) && (
        <p className="muted">Display truncated. Source remains unchanged.</p>
      )}
    </section>
  );
}
export function Dialog({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const titleId = useId();
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    ref.current?.showModal();
    return () => {
      ref.current?.close();
      previous?.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      aria-labelledby={titleId}
    >
      <header className="panel-header">
        <h2 id={titleId}>{title}</h2>
        <button aria-label="Close dialog" onClick={onClose}>
          Close
        </button>
      </header>
      {children}
    </dialog>
  );
}
export class ViewBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <p role="alert">
        This view could not display the supplied data. Close it to return to your draft.
      </p>
    ) : (
      this.props.children
    );
  }
}
export function Field({
  label,
  value,
  onCommit,
  multiline = false,
  live = false,
}: {
  label: string;
  value: string;
  onCommit: (value: string, transaction?: string) => void;
  multiline?: boolean;
  live?: boolean;
}) {
  const [buffer, setBuffer] = useState(value);
  const transaction = useRef(randomId());
  useEffect(() => setBuffer(value), [value]);
  const props = {
    value: buffer,
    onFocus: () => {
      transaction.current = randomId();
    },
    onChange: (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setBuffer(event.target.value);
      if (live) onCommit(event.target.value, transaction.current);
    },
    onBlur: () => {
      if (!live && buffer !== value) onCommit(buffer);
    },
  };
  return (
    <label className="field">
      {label}
      {multiline ? (
        <textarea {...props} rows={6} spellCheck={false} />
      ) : (
        <input {...props} type="text" />
      )}
    </label>
  );
}
