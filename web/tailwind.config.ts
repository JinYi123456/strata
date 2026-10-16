import type { Config } from "tailwindcss";

/**
 * The design language, defined once.
 *
 * The palette is an instrument panel rather than a document: near-black ground,
 * a single teal accent for anything the system is confident about, and a
 * separate hue per change kind so a glance at the ledger or the overlay tells
 * you what happened without reading a label.
 *
 * The kind colours are the load-bearing part. Departures and arrivals must be
 * distinguishable at a glance and by someone who cannot rely on hue alone, so
 * every kind also carries its own glyph in the UI.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ground: {
          DEFAULT: "#080B0F",
          raised: "#0E141B",
          panel: "#131B24",
          edge: "#1E2A36",
          hover: "#18222D",
        },
        ink: {
          DEFAULT: "#E9EFF5",
          muted: "#93A3B4",
          faint: "#5C6B7C",
        },
        accent: {
          DEFAULT: "#4FE3C1",
          soft: "#1C3F3A",
          bright: "#7BF0D5",
        },
        kind: {
          missing: "#FF6B6B",
          added: "#4FE3C1",
          moved: "#B18CFF",
          state: "#63B3FF",
          uncertain: "#FFB84D",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Inter",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Cascadia Code",
          "JetBrains Mono",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.06em" }],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 12px 32px -12px rgba(0,0,0,0.7)",
        lift: "0 20px 48px -16px rgba(0,0,0,0.8)",
        glow: "0 0 0 1px rgba(79,227,193,0.35), 0 0 32px -6px rgba(79,227,193,0.28)",
      },
      keyframes: {
        scanline: {
          "0%": { transform: "translateY(-100%)", opacity: "0" },
          "12%": { opacity: "0.85" },
          "88%": { opacity: "0.85" },
          "100%": { transform: "translateY(100%)", opacity: "0" },
        },
        pulseRing: {
          "0%": { transform: "scale(0.85)", opacity: "0.7" },
          "100%": { transform: "scale(1.9)", opacity: "0" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-500px 0" },
          "100%": { backgroundPosition: "500px 0" },
        },
      },
      animation: {
        scanline: "scanline 2.4s cubic-bezier(0.4,0,0.2,1) infinite",
        pulseRing: "pulseRing 2s cubic-bezier(0.4,0,0.2,1) infinite",
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
