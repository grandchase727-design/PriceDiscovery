export const C = {
  bg: "#0a0e17", panel: "#111827", text: "#e5e7eb",
  cyan: "#06b6d4", green: "#22c55e", red: "#ef4444",
  yellow: "#f59e0b", orange: "#f97316", blue: "#3b82f6",
  gray: "#6b7280", purple: "#8b5cf6", brown: "#a87c5a",
};

export const CLASS_COLORS: Record<string, string> = {
  "🟢 CONTINUATION": C.green,
  "🔵 RECOVERY": C.blue,
  "🟣 COUNTER_RALLY": C.purple,
  "🟡 CONSOLIDATION": C.yellow,
  "🟠 NEUTRAL": C.orange,
  "🟤 FADING": C.brown,
  "🔶 PULLBACK": "#ff8c00",
  "⚠️ WEAKENING": "#dc2626",
  "⬇️ DOWNTREND": C.red,
  "🟡 OVEREXTENDED": C.yellow,
  "🔵 FORMATION": "#60a5fa",
  "🟤 EXHAUSTING": C.brown,
  "🔴 CYCLE_PEAK": "#dc143c",
};

export const DARK_LAYOUT: any = {
  paper_bgcolor: C.bg,
  plot_bgcolor: C.panel,
  font: { color: C.text, size: 11 },
  margin: { t: 40, b: 40, l: 50, r: 20 },
};
