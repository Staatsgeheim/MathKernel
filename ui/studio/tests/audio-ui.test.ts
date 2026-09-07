// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { renderAudio } from '../src/viewers/audioFrame';
const data = {
  schema: 'studio-audio/1' as const,
  artifact: 'audio-preview' as const,
  title: 'Audition',
  events: [{ label: 'one', time: 0, duration: 3, frequency: 440, gain: 1, pan: 0 }],
};
afterEach(() => {
  window.dispatchEvent(new Event('pagehide'));
  document.body.replaceChildren();
  vi.unstubAllGlobals();
});
it('does not create audio before Play, and releases the context on Pause and view close', async () => {
  const close = vi.fn(async () => {}),
    start = vi.fn(),
    stop = vi.fn();
  const graphNode = () => ({
    connect: vi.fn().mockReturnThis(),
    disconnect: vi.fn(),
    gain: { value: 0 },
    pan: { value: 0 },
  });
  const create = vi.fn();
  class Audio {
    currentTime = 0;
    sampleRate = 48000;
    destination = {};
    constructor() {
      create();
    }
    resume = async () => {};
    close = close;
    createGain = graphNode;
    createStereoPanner = graphNode;
    createOscillator = () => ({ ...graphNode(), frequency: { value: 0 }, start, stop });
  }
  vi.stubGlobal('AudioContext', Audio);
  const send = vi.fn();
  renderAudio(data, send);
  expect(create).not.toHaveBeenCalled();
  expect(send).toHaveBeenCalledWith('ready', expect.stringContaining('Playback is stopped'));
  fireEvent.click(screen.getByRole('button', { name: 'Play' }));
  await waitFor(() => expect(start).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole('button', { name: 'Mute' }));
  expect(screen.getByRole('button', { name: 'Unmute' }).getAttribute('aria-pressed')).toBe('true');
  fireEvent.click(screen.getByRole('button', { name: 'Pause' }));
  expect(close).toHaveBeenCalledTimes(1);
  expect(stop).toHaveBeenCalled();
  fireEvent.input(screen.getByLabelText('Position in seconds'), { target: { value: '1.5' } });
  expect(screen.getByRole('status').textContent).toContain('1.50');
  fireEvent.click(screen.getByRole('button', { name: 'Play' }));
  await waitFor(() => expect(create).toHaveBeenCalledTimes(2));
  window.dispatchEvent(new Event('pagehide'));
  expect(close).toHaveBeenCalledTimes(2);
});
it('unsupported device failure leaves a text alternative', async () => {
  vi.stubGlobal(
    'AudioContext',
    class {
      constructor() {
        throw new Error('No device');
      }
    },
  );
  renderAudio(data, vi.fn());
  fireEvent.click(screen.getByRole('button', { name: 'Play' }));
  await waitFor(() =>
    expect(screen.getByRole('status').textContent).toContain('Audio unavailable'),
  );
  expect(screen.getByRole('table').textContent).toContain('440');
  expect((screen.getByRole('button', { name: 'Play' }) as HTMLButtonElement).disabled).toBe(false);
});
