// Design tokens ported from the handoff (wireframe-shared.jsx -> WF).
export const WF = {
  bg: "#fafaf9",
  panel: "#ffffff",
  panelAlt: "#f5f5f4",
  greybox: "#e7e5e4",
  greyboxBorder: "#d6d3d1",
  divider: "#e7e5e4",
  border: "#d6d3d1",
  borderStrong: "#a8a29e",
  text: "#1c1917",
  textMute: "#57534e",
  textDim: "#a8a29e",
  accent: "oklch(0.55 0.13 250)",
  accentBg: "oklch(0.95 0.03 250)",
  accentBorder: "oklch(0.82 0.07 250)",
  ok: "oklch(0.55 0.10 145)",
  okBg: "oklch(0.95 0.04 145)",
  warn: "oklch(0.62 0.13 60)",
  warnBg: "oklch(0.95 0.05 80)",
  sans: '-apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif',
  mono: 'ui-monospace, "SF Mono", Monaco, "Cascadia Code", Consolas, monospace',
} as const;

// Turn an asset reference (relative "assets/..", https URL, or data URI) into a
// URL the browser can load. Relative refs are served by the backend at /assets.
export function assetUrl(ref: string): string {
  if (/^(https?:|data:)/.test(ref)) return ref;
  return "/" + ref.replace(/^\/+/, "");
}
