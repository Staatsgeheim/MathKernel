import type { AudioData } from './audio';

export function renderAudio(
  data: AudioData,
  send: (kind: 'ready' | 'selection' | 'error', text: string) => void,
) {
  const title = document.createElement('h2');
  title.textContent = data.title;
  const note = document.createElement('p');
  note.textContent =
    'Sine-event audition, not the exported PCM waveform. No mapping, normalization or analysis is rerun. Gain is divided by event count to bound summed amplitude; volume starts at 10%. Source values remain unchanged.';
  const status = document.createElement('p');
  status.setAttribute('role', 'status');
  status.textContent = 'Stopped. Playback requires Play.';
  const controls = document.createElement('div');
  const play = document.createElement('button');
  play.textContent = 'Play';
  const pause = document.createElement('button');
  pause.textContent = 'Pause';
  pause.disabled = true;
  const stop = document.createElement('button');
  stop.textContent = 'Stop';
  const volumeLabel = document.createElement('label');
  volumeLabel.textContent = 'Volume ';
  const volume = document.createElement('input');
  volume.type = 'range';
  volume.min = '0';
  volume.max = '1';
  volume.step = '.01';
  volume.value = '.1';
  volumeLabel.append(volume);
  const mute = document.createElement('button');
  mute.textContent = 'Mute';
  mute.setAttribute('aria-pressed', 'false');
  const seekLabel = document.createElement('label');
  seekLabel.textContent = 'Position in seconds ';
  const duration = Math.max(...data.events.map((e) => e.time + e.duration));
  const seek = document.createElement('input');
  seek.type = 'range';
  seek.min = '0';
  seek.max = String(duration);
  seek.step = '.01';
  seek.value = '0';
  seekLabel.append(seek);
  let context: AudioContext | null = null,
    master: GainNode | null = null,
    nodes: OscillatorNode[] = [];
  let base = 0,
    offset = 0,
    muted = false,
    generation = 0;
  function position() {
    return context ? Math.min(duration, offset + context.currentTime - base) : offset;
  }
  function release(reset = false) {
    offset = reset ? 0 : position();
    generation++;
    nodes.forEach((n) => {
      try {
        n.stop();
      } catch {
        /* Already stopped */
      }
      n.disconnect();
    });
    nodes = [];
    const old = context;
    context = null;
    master = null;
    if (old) void old.close().catch(() => {});
    play.disabled = false;
    pause.disabled = true;
    seek.value = String(offset);
  }
  play.onclick = async () => {
    release();
    if (offset >= duration) offset = 0;
    const token = generation;
    try {
      const ctx = new AudioContext();
      context = ctx;
      await ctx.resume();
      if (token !== generation) {
        void ctx.close().catch(() => {});
        return;
      }
      master = ctx.createGain();
      master.gain.value = muted ? 0 : Number(volume.value);
      master.connect(ctx.destination);
      base = ctx.currentTime;
      for (const e of data.events) {
        if (e.time + e.duration <= offset) continue;
        if (e.frequency >= ctx.sampleRate / 2)
          throw new Error('Frequency exceeds this audio device’s Nyquist limit.');
        const oscillator = ctx.createOscillator(),
          gain = ctx.createGain(),
          pan = ctx.createStereoPanner();
        oscillator.type = 'sine';
        oscillator.frequency.value = e.frequency;
        gain.gain.value = e.gain / data.events.length;
        pan.pan.value = e.pan;
        oscillator.connect(gain).connect(pan).connect(master);
        oscillator.start(base + Math.max(0, e.time - offset));
        oscillator.stop(base + e.time + e.duration - offset);
        nodes.push(oscillator);
      }
      play.disabled = true;
      pause.disabled = false;
      status.textContent = 'Playing audition.';
    } catch {
      release();
      status.textContent =
        'Audio unavailable or device constraints unsupported. Use the event table.';
      send('error', status.textContent);
    }
  };
  pause.onclick = () => {
    release();
    status.textContent = 'Paused. Play resumes from the selected position.';
  };
  stop.onclick = () => {
    release(true);
    status.textContent = 'Stopped.';
  };
  seek.oninput = () => {
    const requested = Number(seek.value);
    release();
    offset = requested;
    seek.value = String(offset);
    status.textContent = `Paused at ${offset.toFixed(2)} seconds. Press Play to audition.`;
  };
  volume.oninput = () => {
    if (master) master.gain.value = muted ? 0 : Number(volume.value);
  };
  mute.onclick = () => {
    muted = !muted;
    mute.textContent = muted ? 'Unmute' : 'Mute';
    mute.setAttribute('aria-pressed', String(muted));
    if (master) master.gain.value = muted ? 0 : Number(volume.value);
  };
  const timer = window.setInterval(() => {
    if (context) {
      seek.value = String(position());
      if (position() >= duration) {
        release(true);
        status.textContent = 'Audition finished.';
      }
    }
  }, 100);
  controls.append(play, pause, stop, volumeLabel, mute, seekLabel);
  const table = document.createElement('table');
  const caption = document.createElement('caption');
  caption.textContent =
    'Source event alternative (first 100 events; full source remains in the parent inspector)';
  table.append(caption);
  const header = table.createTHead().insertRow();
  ['Label', 'Time (s)', 'Duration (s)', 'Frequency (Hz)', 'Gain', 'Pan'].forEach((name) => {
    const cell = document.createElement('th');
    cell.scope = 'col';
    cell.textContent = name;
    header.append(cell);
  });
  const body = table.createTBody();
  data.events.slice(0, 100).forEach((e) => {
    const row = body.insertRow();
    [e.label, e.time, e.duration, e.frequency, e.gain, e.pan].forEach((v) => {
      row.insertCell().textContent = String(v);
    });
  });
  document.body.replaceChildren(title, note, controls, status, table);
  window.addEventListener(
    'pagehide',
    () => {
      release(true);
      clearInterval(timer);
    },
    { once: true },
  );
  send('ready', `${data.events.length} audio events ready. Playback is stopped.`);
}
