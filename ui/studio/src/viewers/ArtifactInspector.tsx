import { randomId } from '../security/identity';
import { useEffect, useRef, useState } from 'react';
import { JsonView, Text } from '../app/components';
import { projectVisualization, viewerMessage, type ViewerData } from './contracts';
import { downloadJson } from '../editor/CompositionPanel';

export function ArtifactInspector({ value }: { value: unknown }) {
  const [active, setActive] = useState<ViewerData | null>(null);
  const obj = value && typeof value === 'object' ? (value as Record<string, unknown>) : {};
  const wrapped =
    obj.artifact_schema === 'mathkernel-artifact/1.0' &&
    obj.schema_version === '1.0' &&
    Array.isArray(obj.visualizations)
      ? obj.visualizations.slice(0, 20)
      : [];
  const children = wrapped.map(projectVisualization).filter((v) => v !== null);
  const projected = wrapped.length
    ? {
        title: String(obj.title ?? 'Scientific artifact'),
        views: children.flatMap((v, i) =>
          v.views.map((view) => ({ ...view, artifact: `v${i}-${view.artifact}`.slice(0, 128) })),
        ),
        unsupported: children.flatMap((v) => v.unsupported),
        lineage: {
          assumptions: obj.assumptions,
          transformations: obj.transformations,
          reproducibility: obj.reproducibility,
          visualizations: children.map((v) => v.lineage),
        },
      }
    : projectVisualization(value);
  if (!projected) return null;
  return (
    <section>
      <h3>
        <Text>{projected.title}</Text>
      </h3>
      <p className="notice">
        Presentation of existing data. Screen coordinates use floating point; the source remains
        unchanged. Viewer labels never admit evidence.
      </p>
      {projected.views.map((v) => (
        <button key={v.artifact} onClick={() => setActive(v)}>
          Open {v.title}
        </button>
      ))}
      {projected.unsupported.map((v, i) => (
        <p className="muted" key={i}>
          <Text>{v}</Text>
        </p>
      ))}
      {active && (
        <>
          <button onClick={() => setActive(null)}>Close active viewer</button>
          <IsolatedViewer key={active.artifact} data={active} />
        </>
      )}
      <details>
        <summary>Source transformations and provenance</summary>
        <JsonView value={projected.lineage} />
      </details>
      <button onClick={() => downloadJson(value, 'existing-visualization.json')}>
        Export existing source representation
      </button>
    </section>
  );
}
function IsolatedViewer({ data }: { data: ViewerData }) {
  const frame = useRef<HTMLIFrameElement>(null),
    port = useRef<MessagePort | null>(null);
  const [status, setStatus] = useState('Opening isolated viewer…');
  const [channel] = useState(() => randomId());
  useEffect(() => {
    const initialize = (event: MessageEvent) => {
      if (event.source !== frame.current?.contentWindow || event.data?.kind !== 'studio-view-ready')
        return;
      port.current?.close();
      const message = new MessageChannel();
      port.current = message.port1;
      let sequence = 0;
      message.port1.onmessage = (reply) => {
        const v = viewerMessage.safeParse(reply.data);
        if (
          !v.success ||
          v.data.channel !== channel ||
          v.data.artifact !== data.artifact ||
          v.data.seq <= sequence
        )
          return;
        sequence = v.data.seq;
        setStatus(v.data.text);
      };
      frame.current?.contentWindow?.postMessage({ kind: 'studio-view-init', channel, data }, '*', [
        message.port2,
      ]);
      setStatus('Renderer ready; waiting for its acknowledgement.');
    };
    window.addEventListener('message', initialize);
    const timer = setTimeout(
      () =>
        setStatus((old) =>
          old.includes('points displayed')
            ? old
            : 'The isolated renderer did not start. Close this viewer and use the source representation below; no computation or network fallback was requested.',
        ),
      8000,
    );
    return () => {
      window.removeEventListener('message', initialize);
      clearTimeout(timer);
      port.current?.close();
    };
  }, [channel, data]);
  return (
    <>
      <p role="status">
        <Text>{status}</Text>
      </p>
      <iframe
        ref={frame}
        title={`Isolated presentation: ${data.title}`}
        src="/studio/viewer.html"
        sandbox="allow-scripts"
        referrerPolicy="no-referrer"
        className="artifact-frame"
        onLoad={() => setStatus('Frame loaded; waiting for renderer readiness.')}
      />
    </>
  );
}
