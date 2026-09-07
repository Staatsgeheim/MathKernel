import { expect, it } from 'vitest';
import { projectVisualization, viewerSchema, viewerMessage } from '../src/viewers/contracts';
const visual = {
  artifact_schema: 'mathkernel-viz/2.0',
  schema_version: '2.0',
  title: '<script>hostile</script>',
  blocks: [
    {
      block_id: 'plot',
      kind: 'plot2d',
      title: 'Coordinates',
      series: ['s'],
      bindings: {},
      config: {},
    },
  ],
  series: { s: { x: 'x', y: 'y', label: 's' } },
  datasets: { x: { encoding: 'inline', data: [0, 1] }, y: { encoding: 'inline', data: [1, 2] } },
};
it('projects only existing inline points without propagating result or authority labels', () => {
  const p = projectVisualization({ ...visual, trust: 'formal', approved: true })!;
  expect(p.views[0]!.series[0]!.points).toEqual([
    [0, 1],
    [1, 2],
  ]);
  expect(p.views[0]).not.toHaveProperty('trust');
  expect(p.views[0]).not.toHaveProperty('approved');
});
it('unsupported transforms and compressed datasets are inert fallbacks, not silently changed pictures', () => {
  expect(
    projectVisualization({ ...visual, blocks: [{ ...visual.blocks[0], config: { log_y: true } }] })!
      .views,
  ).toHaveLength(0);
  expect(
    projectVisualization({
      ...visual,
      datasets: { ...visual.datasets, x: { encoding: 'zlib+base64', data: 'hostile' } },
    })!.unsupported,
  ).toHaveLength(1);
});
it('rejects unknown schemas and oversized active payloads', () => {
  expect(projectVisualization({ ...visual, artifact_schema: 'mathkernel-viz/999' })).toBeNull();
  const v = projectVisualization(visual)!.views[0]!;
  expect(
    viewerSchema.safeParse({
      ...v,
      series: [{ label: 'large', points: Array.from({ length: 10001 }, () => [0, 0]) }],
    }).success,
  ).toBe(false);
});
it('viewer messages cannot express a host command or carry arbitrary fields', () => {
  const message = {
    channel: crypto.randomUUID(),
    artifact: 'plot',
    seq: 1,
    kind: 'ready',
    text: 'Ready',
  };
  expect(viewerMessage.safeParse(message).success).toBe(true);
  expect(viewerMessage.safeParse({ ...message, kind: 'submit' }).success).toBe(false);
  expect(viewerMessage.safeParse({ ...message, authority: 'forged' }).success).toBe(false);
});
