import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Focused Vitest config for the M6.3 replay component/integration tests.
// The dev-server proxy is irrelevant here: the API client is mocked.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.tsx"],
    setupFiles: ["./src/test/setup.ts"],
  },
});
