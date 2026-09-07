import { viewerSchema } from './contracts';
let initialized = false;
const announce = () => parent.postMessage({ kind: 'studio-view-ready' }, '*');
const readiness = window.setInterval(() => {
  if (initialized) window.clearInterval(readiness);
  else announce();
}, 100);
window.addEventListener('message', (event) => {
  if (
    initialized ||
    event.source !== parent ||
    event.data?.kind !== 'studio-view-init' ||
    !/^[0-9a-f-]{36}$/.test(event.data.channel ?? '') ||
    event.ports.length !== 1
  )
    return;
  initialized = true;
  window.clearInterval(readiness);
  const port = event.ports[0]!,
    channel = event.data.channel;
  const parsed = viewerSchema.safeParse(event.data.data);
  if (!parsed.success) {
    document.body.textContent =
      'Presentation payload does not match the supported viewer contract.';
    port.close();
    return;
  }
  const data = parsed.data;
  let seq = 0;
  const send = (kind: 'ready' | 'selection', text: string) =>
    port.postMessage({ channel, artifact: data.artifact, seq: ++seq, kind, text });
  const title = document.createElement('h2');
  title.textContent = data.title;
  const description = document.createElement('p');
  description.textContent =
    'Existing coordinates only. Drag-free rotation controls change the view, not the source.';
  const canvas = document.createElement('canvas');
  canvas.width = 720;
  canvas.height = 400;
  canvas.setAttribute('aria-label', data.title);
  canvas.setAttribute('role', 'img');
  const controls = document.createElement('div');
  let angle = 0;
  const points = data.series.flatMap((s) => s.points);
  function draw() {
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, 720, 400);
    ctx.fillStyle = '#111b2a';
    ctx.fillRect(0, 0, 720, 400);
    if (!points.length) return;
    const project = (p: number[]) =>
      data.kind === 'plot2d'
        ? [p[0]!, p[1]!]
        : [p[0]! * Math.cos(angle) + (p[2] ?? 0) * Math.sin(angle), p[1]!];
    const projected = points.map(project);
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of projected) {
      minX = Math.min(minX, p[0]!);
      maxX = Math.max(maxX, p[0]!);
      minY = Math.min(minY, p[1]!);
      maxY = Math.max(maxY, p[1]!);
    }
    ctx.fillStyle = '#dce8f7';
    ctx.font = '14px system-ui';
    ctx.fillText(`x: ${minX.toPrecision(4)} … ${maxX.toPrecision(4)}`, 30, 390);
    ctx.fillText(`y: ${minY.toPrecision(4)} … ${maxY.toPrecision(4)}`, 30, 20);
    const colors = ['#63dec8', '#ffc578', '#a8baff', '#ff9ea9'];
    data.series.forEach((s, i) => {
      ctx.strokeStyle = colors[i % colors.length]!;
      ctx.fillStyle = ctx.strokeStyle;
      ctx.lineWidth = 2;
      ctx.beginPath();
      s.points.forEach((p, j) => {
        const xy = project(p),
          x = 35 + ((xy[0]! - minX) / (maxX - minX || 1)) * 650,
          y = 355 - ((xy[1]! - minY) / (maxY - minY || 1)) * 315;
        if (data.kind === 'point_cloud_3d') ctx.fillRect(x - 2, y - 2, 4, 4);
        else if (!j) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (data.kind !== 'point_cloud_3d') ctx.stroke();
    });
  }
  if (data.kind !== 'plot2d')
    for (const [label, delta] of [
      ['Rotate left', -0.15],
      ['Rotate right', 0.15],
    ] as const) {
      const button = document.createElement('button');
      button.textContent = label;
      button.onclick = () => {
        angle += delta;
        draw();
      };
      controls.append(button);
    }
  const table = document.createElement('table'),
    caption = document.createElement('caption');
  caption.textContent =
    'Coordinate text alternative — first 100 points; full source remains in the parent inspector';
  table.append(caption);
  const body = document.createElement('tbody');
  for (const [index, p] of points.slice(0, 100).entries()) {
    const row = document.createElement('tr'),
      number = document.createElement('th');
    number.textContent = String(index + 1);
    row.append(number);
    for (const value of p) {
      const cell = document.createElement('td');
      cell.textContent = String(value);
      row.append(cell);
    }
    body.append(row);
  }
  table.append(body);
  const legend = document.createElement('p');
  legend.textContent = data.series.map((s) => s.label).join(' · ');
  document.body.replaceChildren(title, description, controls, canvas, legend, table);
  draw();
  send('ready', `${points.length} points displayed in an isolated fixed renderer.`);
  window.addEventListener('pagehide', () => port.close(), { once: true });
});
announce();
window.setTimeout(() => window.clearInterval(readiness), 8000);
