import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Fonts are bundled (no Google Fonts CDN) so the site renders the same offline.
import "@fontsource/fredoka/latin-600.css";
import "@fontsource/fredoka/latin-700.css";
import "@fontsource/nunito/latin-500.css";
import "@fontsource/nunito/latin-700.css";
import "@fontsource/dm-mono/latin-500.css";
import "./styles.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

// Offline support for the demo (project.md §6, criterion 5): the worker is only built for
// production (see precache.ts), so dev keeps normal live reload.
if (import.meta.env.PROD && "serviceWorker" in navigator) {
  const base = import.meta.env.BASE_URL;
  navigator.serviceWorker.register(`${base}sw.js`, { scope: base }).catch((err) => {
    console.error("Trend Court: offline cache unavailable", err);
  });
}
