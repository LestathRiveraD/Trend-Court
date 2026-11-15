/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { precache } from "./precache";

// GitHub Pages serves the repo at https://<user>.github.io/Trend-Court/, so every
// asset URL (and the case JSON, via import.meta.env.BASE_URL) lives under /Trend-Court/.
// Dev and preview use the same base, so a path that works locally works on Pages.
const base = "/Trend-Court/";

export default defineConfig({
  base,
  // precache() writes dist/sw.js so the built site keeps working offline after one visit.
  plugins: [react(), precache()],
  build: {
    // Plotly alone is ~4.6 MB minified; it is split into its own lazily loaded chunk.
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: "node",
    // Vitest serves from "/" by default; run the loaders under the real Pages base.
    env: { BASE_URL: base },
    include: ["src/**/*.test.ts", "*.test.ts"],
  },
});
