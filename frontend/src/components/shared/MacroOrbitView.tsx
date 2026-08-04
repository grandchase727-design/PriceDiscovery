/**
 * MacroOrbitView — 3D "solar system" of the Macro Regime FRED indicators.
 * Each indicator = a planet orbiting the neutral-regime sun. Mapping (user-chosen):
 *   • angle  = TIME  (one full orbit = the trailing window, oldest→newest)
 *   • radius = z-score (eccentric swing: outward when z high, inward when z low)
 *   • height = z (planets rise/dip out of the ecliptic for 3D depth)
 * Colored by axis (성장/인플/크레딧/유동성). Play/scrub replays the history; the
 * comet trail shows the recent path. Deliberately single dark "space" theme.
 * Data: /api/macro-regime/orbit — per-indicator monthly rolling-z time series.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars, Line, Html } from "@react-three/drei";
import * as THREE from "three";
import { fetchMacroOrbit } from "../../api/client";
import { C } from "../../styles/theme";

const AXIS: Record<string, { color: string; ko: string }> = {
  growth: { color: "#22C55E", ko: "성장" },
  inflation: { color: "#F5A34A", ko: "인플" },
  credit: { color: "#5AA6E8", ko: "크레딧" },
  liquidity: { color: "#EC6A9C", ko: "유동성" },
};
const AXIS_ORDER = ["growth", "inflation", "credit", "liquidity"];

// forward-fill nulls (release lag); lead nulls take the first real value
function ffill(z: (number | null)[]): number[] {
  const first = z.find((v) => v != null);
  let last = first == null ? 0 : first;
  return z.map((v) => (v == null ? last : (last = v)));
}

// full orbit loop: angle=time (closed loop over the window), radius/height = z
function buildLoop(zf: number[], baseR: number, phase: number, amp: number): THREE.Vector3[] {
  const N = zf.length;
  const pts: THREE.Vector3[] = [];
  for (let j = 0; j < N; j++) {
    const theta = phase + (j / (N - 1)) * Math.PI * 2;
    const r = Math.max(0.5, baseR + zf[j] * amp);
    pts.push(new THREE.Vector3(r * Math.cos(theta), zf[j] * 0.28, r * Math.sin(theta)));
  }
  return pts;
}

function Driver({ playing, speed, N, progressRef, setCursor }: {
  playing: boolean; speed: number; N: number;
  progressRef: React.MutableRefObject<number>; setCursor: (i: number) => void;
}) {
  const lastInt = useRef(-1);
  useFrame((_, dt) => {
    if (playing) progressRef.current = (progressRef.current + dt * speed) % N;
    const fi = Math.floor(progressRef.current);
    if (fi !== lastInt.current) { lastInt.current = fi; setCursor(fi); }
  });
  return null;
}

function Planet({ loop, color, code, progressRef, trailCursor }: {
  loop: THREE.Vector3[]; color: string; code: string;
  progressRef: React.MutableRefObject<number>; trailCursor: number;
}) {
  const g = useRef<THREE.Group>(null);
  const N = loop.length;
  useFrame(() => {
    if (!g.current) return;
    const p = progressRef.current;
    const i = Math.floor(p) % N;
    const f = p - Math.floor(p);
    g.current.position.lerpVectors(loop[i], loop[(i + 1) % N], f);
  });
  const trail = useMemo(() => {
    const K = 12, arr: THREE.Vector3[] = [];
    for (let k = K; k >= 0; k--) arr.push(loop[(((trailCursor - k) % N) + N) % N]);
    return arr;
  }, [trailCursor, loop, N]);
  return (
    <>
      <Line points={loop} color={color} lineWidth={1} transparent opacity={0.16} />
      <Line points={trail} color={color} lineWidth={2.5} transparent opacity={0.9} />
      <group ref={g}>
        <mesh>
          <sphereGeometry args={[0.17, 20, 20]} />
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.7} />
        </mesh>
        <Html center distanceFactor={16} style={{ pointerEvents: "none" }}>
          <span style={{ color: "#fff", fontSize: 11, fontWeight: 700, whiteSpace: "nowrap",
            textShadow: "0 0 5px #000, 0 0 2px #000" }}>{code}</span>
        </Html>
      </group>
    </>
  );
}

function Scene({ planets, cursor, setCursor, playing, speed, progressRef }: any) {
  const N = planets[0]?.z?.length ?? 0;
  const built = useMemo(() => {
    const sorted = [...planets].sort(
      (a: any, b: any) => AXIS_ORDER.indexOf(a.axis) - AXIS_ORDER.indexOf(b.axis));
    return sorted.map((p: any, i: number) => ({
      ...p,
      loop: buildLoop(ffill(p.z), 2.6 + i * 0.85, (i / sorted.length) * Math.PI * 2, 0.5),
    }));
  }, [planets]);
  return (
    <>
      <ambientLight intensity={0.45} />
      <pointLight position={[0, 0, 0]} intensity={2.6} distance={44} color="#FFE6A8" />
      <mesh>
        <sphereGeometry args={[0.85, 32, 32]} />
        <meshStandardMaterial color="#FFCA55" emissive="#FFAE22" emissiveIntensity={1.5} />
      </mesh>
      <Stars radius={90} depth={45} count={2600} factor={3.2} fade speed={0.4} />
      {built.map((p: any) => (
        <Planet key={p.code} loop={p.loop} color={(AXIS[p.axis] || AXIS.credit).color}
          code={p.code} progressRef={progressRef} trailCursor={cursor} />
      ))}
      <Driver playing={playing} speed={speed} N={N} progressRef={progressRef} setCursor={setCursor} />
      <OrbitControls enablePan={false} minDistance={6} maxDistance={44}
        autoRotate={!playing} autoRotateSpeed={0.35} />
    </>
  );
}

export default function MacroOrbitView({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(9);
  const progressRef = useRef(0);

  useEffect(() => {
    let alive = true;
    fetchMacroOrbit(54)
      .then((r) => { if (alive) { setData(r); const n = r?.dates?.length ?? 1; progressRef.current = n - 1; setCursor(n - 1); } })
      .catch((e: any) => { if (alive) setErr(e?.message || String(e)); });
    return () => { alive = false; };
  }, []);

  const N = data?.dates?.length ?? 0;
  const curDate = data?.dates?.[cursor] ?? "";
  const planets = data?.planets ?? [];
  const legend = AXIS_ORDER.filter((ax) => planets.some((p: any) => p.axis === ax));

  const btn: React.CSSProperties = {
    background: "rgba(255,255,255,0.1)", border: "1px solid rgba(255,255,255,0.25)",
    color: "#fff", borderRadius: 6, padding: "3px 10px", fontSize: 12, cursor: "pointer",
  };

  return (
    <div style={{ position: "relative", height: 480, background: "#070B16",
      borderRadius: 8, overflow: "hidden", border: "1px solid #1B2333" }}>
      {!data && !err && <div style={{ position: "absolute", inset: 0, display: "grid",
        placeItems: "center", color: "#8892A6", fontSize: 13 }}>궤도 데이터 로딩 중… (FRED z-이력)</div>}
      {err && <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center",
        color: "#EC6A9C", fontSize: 13 }}>Error: {err}</div>}

      {data && !data.error && N > 0 && (
        <>
          <Canvas camera={{ position: [0, 10, 18], fov: 48 }} gl={{ antialias: true }}>
            <Scene planets={planets} cursor={cursor} setCursor={setCursor}
              playing={playing} speed={speed} progressRef={progressRef} />
          </Canvas>

          {/* title + legend (top-left) */}
          <div style={{ position: "absolute", top: 10, left: 12, pointerEvents: "none" }}>
            <div style={{ color: "#fff", fontSize: 13, fontWeight: 800, textShadow: "0 0 6px #000" }}>
              🪐 Regime Orbit — 경기지표 z-궤적
            </div>
            <div style={{ marginTop: 4, display: "flex", gap: 8, flexWrap: "wrap" }}>
              {legend.map((ax) => (
                <span key={ax} style={{ fontSize: 10.5, color: "#C7CEDA", display: "flex", alignItems: "center", gap: 3 }}>
                  <span style={{ width: 8, height: 8, borderRadius: "50%", background: AXIS[ax].color, display: "inline-block" }} />
                  {AXIS[ax].ko}
                </span>
              ))}
            </div>
          </div>

          {/* close (top-right) */}
          <button type="button" onClick={onClose} style={{ ...btn, position: "absolute", top: 10, right: 12 }}>✕ 닫기</button>

          {/* controls (bottom) */}
          <div style={{ position: "absolute", left: 12, right: 12, bottom: 10, display: "flex",
            alignItems: "center", gap: 10, background: "rgba(7,11,22,0.72)", borderRadius: 8,
            padding: "7px 10px", backdropFilter: "blur(3px)" }}>
            <button type="button" style={btn} onClick={() => setPlaying((v) => !v)}>{playing ? "⏸" : "▶"}</button>
            <span className="mono" style={{ color: "#FFE6A8", fontSize: 13, fontWeight: 700, minWidth: 62 }}>{curDate}</span>
            <input type="range" min={0} max={N - 1} value={cursor}
              onChange={(e) => { const v = +e.target.value; setCursor(v); progressRef.current = v; setPlaying(false); }}
              style={{ flex: 1, accentColor: "#F5A34A" }} />
            <span style={{ color: "#8892A6", fontSize: 11 }}>속도</span>
            <input type="range" min={3} max={20} value={speed} onChange={(e) => setSpeed(+e.target.value)}
              style={{ width: 66, accentColor: "#5AA6E8" }} />
          </div>

          {/* hint */}
          <div style={{ position: "absolute", bottom: 52, right: 14, color: "#5A6478", fontSize: 10, pointerEvents: "none" }}>
            드래그=회전 · 휠=줌 · 각도=시간 · 반경/높이=z
          </div>
        </>
      )}
      {data?.error && <div style={{ position: "absolute", inset: 0, display: "grid",
        placeItems: "center", color: "#EC6A9C", fontSize: 13 }}>{data.error}</div>}
    </div>
  );
}
