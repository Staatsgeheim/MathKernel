import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/studio/',
  build: { outDir: 'host/src/mathkernel_studio/assets', emptyOutDir: true, sourcemap: false },
  worker: { format: 'es' },
});
