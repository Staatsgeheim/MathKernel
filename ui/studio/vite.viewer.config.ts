import { defineConfig } from 'vite';
export default defineConfig({
  publicDir: false,
  build: {
    outDir: 'host/src/mathkernel_studio/assets/viewer',
    emptyOutDir: true,
    lib: {
      entry: 'src/viewers/frame.ts',
      name: 'StudioViewer',
      formats: ['iife'],
      fileName: () => 'viewer.js',
    },
  },
});
