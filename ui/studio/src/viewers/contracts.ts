import { z } from '../security/schema';
import { id } from '../editor/document';
export const viewerSchema = z
  .strictObject({
    schema: z.literal('studio-view/1'),
    artifact: id,
    title: z.string().max(8192),
    kind: z.enum(['plot2d', 'point_cloud_3d', 'trajectory_3d']),
    series: z
      .array(
        z.strictObject({
          label: z.string().max(8192),
          points: z
            .array(z.array(z.number().finite().min(-1e100).max(1e100)).min(2).max(3))
            .max(10000),
        }),
      )
      .max(20),
  })
  .refine((v) => v.series.reduce((n, s) => n + s.points.length, 0) <= 10000);
export type ViewerData = z.infer<typeof viewerSchema>;
export const viewerMessage = z.strictObject({
  channel: z.string().uuid(),
  artifact: id,
  seq: z.number().int().min(1).max(Number.MAX_SAFE_INTEGER),
  kind: z.enum(['ready', 'selection', 'error']),
  text: z.string().max(1000),
});

/** Only inline, already-produced numeric data is projected. No FFT, embedding or sampling. */
export function projectVisualization(
  value: unknown,
): { title: string; views: ViewerData[]; unsupported: string[]; lineage: unknown } | null {
  if (!value || typeof value !== 'object') return null;
  const v = value as Record<string, unknown>;
  if (
    v.artifact_schema !== 'mathkernel-viz/2.0' ||
    v.schema_version !== '2.0' ||
    !Array.isArray(v.blocks) ||
    v.blocks.length > 100
  )
    return null;
  const record = (x: unknown): Record<string, unknown> =>
    x && typeof x === 'object' && !Array.isArray(x) ? (x as Record<string, unknown>) : {};
  const series = record(v.series),
    datasets = record(v.datasets);
  const inline = (ref: unknown) => {
    const d = record(datasets[String(ref)]);
    return d.encoding === 'inline' && Array.isArray(d.data) && d.data.length <= 30000
      ? d.data
      : null;
  };
  const views: ViewerData[] = [],
    unsupported: string[] = [];
  for (const raw of v.blocks) {
    const b = record(raw),
      config = record(b.config),
      name = String(b.title || b.block_id || 'Block');
    if (
      !['plot2d', 'point_cloud_3d', 'trajectory_3d'].includes(String(b.kind)) ||
      Object.keys(record(b.bindings)).length ||
      Object.keys(config).some((key) => !['x_label', 'y_label', 'show_points'].includes(key)) ||
      !Array.isArray(b.series) ||
      b.series.length > 20
    ) {
      unsupported.push(`${name}: unsupported block/configuration; inspect source.`);
      continue;
    }
    try {
      const projected = b.series.map((ref: unknown) => {
        const s = record(series[String(ref)]),
          style = record(s.style);
        if (Object.keys(style).some((key) => key !== 'color' && key !== 'size'))
          throw new Error('Style changes representation');
        let points = s.points;
        if (!Array.isArray(points)) {
          const x = inline(s.x),
            y = inline(s.y);
          if (!x) throw new Error('Inline coordinates unavailable');
          if (b.kind === 'plot2d') {
            if (!y || x.length !== y.length) throw new Error('Coordinate lengths differ');
            points = x.map((value, i) => [value, y[i]]);
          } else {
            if (x.length % 3) throw new Error('Invalid triples');
            points = Array.from({ length: x.length / 3 }, (_, i) => x.slice(i * 3, i * 3 + 3));
          }
        }
        return { label: String(s.label ?? ref), points };
      });
      views.push(
        viewerSchema.parse({
          schema: 'studio-view/1',
          artifact: b.block_id,
          title: name,
          kind: b.kind,
          series: projected,
        }),
      );
    } catch {
      unsupported.push(
        `${name}: unsupported/oversized coordinates or data encoding; source preserved.`,
      );
    }
  }
  return {
    title: String(v.title ?? 'Visualization'),
    views,
    unsupported,
    lineage: {
      assumptions: v.assumptions,
      transformations: v.transformations,
      provenance: v.provenance,
      manifest: v.manifest,
    },
  };
}
