import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api and /health to the FastAPI backend so the UI
// can run without CORS setup in development. For a production build the same
// prefix works behind a reverse proxy or by setting VITE_API_BASE.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
