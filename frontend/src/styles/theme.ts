// ─────────────────────────────────────────────────────────────────
// Financial Times (ft.com) inspired theme
// ─────────────────────────────────────────────────────────────────
// Signature: FT Pink paper (#FFF1E5), near-black serif headlines,
// restrained accent palette (Oxford blue, claret, teal), hairline rules.
// ─────────────────────────────────────────────────────────────────
export const C = {
  // Surfaces
  bg: "#FFF1E5",       // FT Pink paper (primary background)
  panel: "#FFFFFF",    // white card on pink
  bgAlt: "#FBF2E9",    // soft pink (alt rows / recessed)
  bgSunk: "#F2E5D7",   // darker pink (table headers / inputs)
  // Text
  text: "#33302E",     // FT near-black
  gray: "#66605C",     // FT slate (secondary)
  grayMuted: "#9C9690",// muted/tertiary
  // Borders
  border: "#E6D9CE",   // FT wheat hairline
  borderStrong: "#CCC1B7",
  // Accents (FT brand)
  blue: "#0F5499",     // FT Oxford blue (links / primary)
  claret: "#990F3D",   // FT claret (signature dark red)
  teal: "#0D7680",     // FT teal
  // Finance semantics (readable on light pink)
  green: "#0A7D3F",    // up / positive
  red: "#CC0000",      // down / negative
  yellow: "#B85C00",   // warn / amber
  orange: "#B85C00",
  cyan: "#0D7680",     // teal alias
  purple: "#7D5BA6",
  brown: "#8A6D3B",
};

export const CLASS_COLORS: Record<string, string> = {
  "🟢 CONTINUATION": C.green,
  "🔵 RECOVERY": C.blue,
  "🟣 COUNTER_RALLY": C.purple,
  "🟡 CONSOLIDATION": C.yellow,
  "🟠 NEUTRAL": C.orange,
  "🟤 FADING": C.brown,
  "🔶 PULLBACK": "#C2701C",
  "⚠️ WEAKENING": C.claret,
  "⬇️ DOWNTREND": C.red,
  "🟡 OVEREXTENDED": C.yellow,
  "🔵 FORMATION": "#3A7CA5",
  "🟤 EXHAUSTING": C.brown,
  "🔴 CYCLE_PEAK": C.claret,
};

// Plotly layout — FT light paper (kept name DARK_LAYOUT for import compatibility)
export const DARK_LAYOUT: any = {
  paper_bgcolor: "#FFF1E5",
  plot_bgcolor: "#FFFFFF",
  font: { color: "#33302E", size: 13, family: "Georgia, 'Times New Roman', serif" },
  xaxis: { gridcolor: "#E6D9CE", zerolinecolor: "#CCC1B7", linecolor: "#CCC1B7" },
  yaxis: { gridcolor: "#E6D9CE", zerolinecolor: "#CCC1B7", linecolor: "#CCC1B7" },
  margin: { t: 40, b: 40, l: 50, r: 20 },
};

export const FT_LAYOUT = DARK_LAYOUT;
