import { z } from '../security/schema';
const schema = z.strictObject({
  theme: z.enum(['system', 'light', 'dark']),
  textSize: z.number().min(16).max(24),
  palette: z.boolean(),
  inspector: z.boolean(),
  paletteWidth: z.number().min(200).max(420),
  inspectorWidth: z.number().min(280).max(560),
  reducedMotion: z.boolean(),
});
export type Preferences = z.infer<typeof schema>;
export const defaults: Preferences = {
  theme: 'system',
  textSize: 16,
  palette: true,
  inspector: true,
  paletteWidth: 260,
  inspectorWidth: 330,
  reducedMotion: false,
};
export function loadPreferences(): Preferences {
  try {
    return schema.parse(JSON.parse(localStorage.getItem('mk-studio-preferences-1') ?? 'null'));
  } catch {
    return defaults;
  }
}
export function savePreferences(prefs: Preferences) {
  localStorage.setItem('mk-studio-preferences-1', JSON.stringify(schema.parse(prefs)));
}
