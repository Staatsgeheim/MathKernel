import { z } from '../security/schema';

/** Resolved event audition only; mappings and analysis are never executed in Studio. */
export const audioSchema = z
  .strictObject({
    schema: z.literal('studio-audio/1'),
    artifact: z.literal('audio-preview'),
    title: z.string().max(8192),
    events: z
      .array(
        z.strictObject({
          label: z.string().max(8192),
          time: z.number().min(0).max(60),
          duration: z.number().positive().max(60),
          frequency: z.number().min(20).max(20000),
          gain: z.number().min(0).max(1),
          pan: z.number().min(-1).max(1),
        }),
      )
      .min(1)
      .max(500),
  })
  .refine((v) => v.events.every((e) => e.time + e.duration <= 60));
export type AudioData = z.infer<typeof audioSchema>;
export function projectAudio(value: unknown): { view: AudioData | null; reason: string } | null {
  const d = value as Record<string, unknown> | null;
  if (!d || d.artifact_schema !== 'mathkernel-sonify/1.0' || d.schema_version !== '1.0')
    return null;
  try {
    if (!Array.isArray(d.tracks) || d.tracks.length > 20) throw new Error();
    const events = d.tracks.flatMap((t) => {
      if (
        !Array.isArray(t.events) ||
        t.events.length > 500 ||
        !Array.isArray(t.mapping_refs) ||
        t.mapping_refs.length
      )
        throw new Error();
      return t.events.map(
        (e: { values: Record<string, number>; time: number; duration: number; label: string }) => {
          if (!e.values || Object.keys(e.values).some((k) => !['frequency', 'gain'].includes(k)))
            throw new Error();
          return {
            label: `${t.label}: ${e.label}`,
            time: e.time,
            duration: e.duration,
            frequency: e.values.frequency,
            gain: (e.values.gain ?? 1) * (t.gain ?? 1),
            pan: t.pan ?? 0,
          };
        },
      );
    });
    return {
      view: audioSchema.parse({
        schema: 'studio-audio/1',
        artifact: 'audio-preview',
        title: d.title,
        events,
      }),
      reason: '',
    };
  } catch {
    return {
      view: null,
      reason:
        'Audio source preserved. Audition supports at most 500 resolved sine events over 60 seconds, 20–20,000 Hz, gain 0–1, and no mappings, phase or timbre transforms.',
    };
  }
}
