import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dev server proxies to the FastAPI process so the browser sees one origin
// and the WebSocket upgrade works without any CORS dance. 8000 is
// `server.port` in config.yaml; override it if you moved it.
const API = process.env.FRIDAY_API ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API, changeOrigin: true },
      '/ws': { target: API, ws: true },
    },
  },
  build: {
    // The server mounts web/dist directly; keep the names predictable.
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    // `stubGlobal` is not undone by `restoreAllMocks`; without this, one test's
    // fake `fetch` silently becomes the next test's.
    unstubGlobals: true,
  },
})
