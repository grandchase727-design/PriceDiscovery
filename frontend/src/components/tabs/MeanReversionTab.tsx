import { useEffect, useState, useMemo } from "react";
import { fetchMeanReversion } from "../../api/client";
import { C } from "../../styles/theme";

// ---------------------------------------------------------------------------
// Oversold Reversion — Mean-Reversion tier (OER의 거울상)
// ---------------------------------------------------------------------------
// 모멘텀 게이트가 REJECT한 종목 중, 통계적으로 과매도 + 안정화 중 + 품질 플로어를
// 통과한 dislocation을 별도 tier로 surface. Composite와 완전 분리.
// MR Score = 0.30 OU z-dislocation + 0.20 idiosyncratic residual + 0.18 stabilization
//          + 0.15 long-term reversal(De Bondt-Thaler) + 0.17 mean-distance stretch.
// regime-gated (횡보·고분산장 amplify / 강추세장 suppress), half-life 사이징.
// ---------------------------------------------------------------------------

interface MRCandidate {
  ticker: string; name: string; sector: string; asset_type: string;
  classification: string; mr_score: number; mr_score_adj: number;
  mr_ou: number; mr_idio: number; mr_stab: number; mr_lt: number; mr_stretch: number;
  mr_half_life_days: number; qvr_score: number; composite: number; oer: number;
  rsi: number; range_pct: number; pct_from_high: number;
  sma20_dist: number; sma50_dist: number; ret_5d: number; ret_21d: number;
  realized_vol: number; ret_1m: number; ret_3m: number;
}

function SubBar({ label, v, color }: { label: string; v: number; color: string }) {
  const val = v ?? 50;
  return (
    <div className="flex items-center gap-1" title={`${label}: ${val.toFixed(0)}`}>
      <span className="text-[9px] w-7" style={{ color: C.gray }}>{label}</span>
      <div className="flex-1 h-1.5 rounded" style={{ backgroundColor: C.border, minWidth: 28 }}>
        <div className="h-1.5 rounded" style={{ width: `${Math.max(0, Math.min(100, val))}%`, backgroundColor: color }} />
      </div>
      <span className="text-[9px] w-5 text-right font-mono" style={{ color: C.text }}>{val.toFixed(0)}</span>
    </div>
  );
}

function pctColor(v: number | null | undefined) {
  if (v == null) return C.gray;
  return v > 0 ? C.green : v < 0 ? C.red : C.gray;
}
function fpct(v: number | null | undefined) {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

export function MeanReversionTab() {
  const [data, setData] = useState<MRCandidate[]>([]);
  const [mult, setMult] = useState(0.7);
  const [note, setNote] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [typeFilter, setTypeFilter] = useState<"ALL" | "Stock" | "ETF">("ALL");

  useEffect(() => {
    setLoading(true);
    fetchMeanReversion()
      .then((r) => {
        setData(r.candidates || []);
        setMult(r.regime_multiplier ?? 0.7);
        setNote(r.note || "");
      })
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(
    () => (typeFilter === "ALL" ? data : data.filter((d) => d.asset_type === typeFilter)),
    [data, typeFilter]
  );

  if (loading) return <div className="text-[#857F7A] p-8">Loading oversold-reversion data…</div>;

  const avg = data.length ? data.reduce((s, d) => s + (d.mr_score || 0), 0) / data.length : 0;
  const regimeLabel = mult >= 0.85 ? "🟢 MR 우호 (횡보·고분산)" : mult <= 0.5 ? "🔴 MR 비우호 (강추세)" : "🟡 중립";

  return (
    <div className="space-y-5">
      {/* Intro */}
      <div className="rounded-lg border p-3" style={{ borderColor: C.purple + "55", background: C.purple + "0d" }}>
        <div className="flex items-baseline justify-between mb-1.5">
          <h3 className="text-[16px] font-bold" style={{ color: C.purple }}>
            🔄 Oversold Reversion — Mean-Reversion Tier
          </h3>
          <span className="text-[12px]" style={{ color: C.gray }}>
            OER의 거울상 (과매도→상승 반전) · Composite와 완전 분리
          </span>
        </div>
        <p className="text-[13px] leading-relaxed" style={{ color: C.text }}>
          모멘텀 게이트가 <strong>탈락</strong>시킨 종목 중 통계적으로 과매도이면서 <strong>하락이 소진(안정화)</strong>되고
          품질 플로어(QVR≥45)를 통과한 dislocation만 표시합니다. Falling-knife 방어: 구조적 하락(DOWNTREND/WEAKENING/FADING) 거부 ·
          fresh-break(오늘 급락+지속) 차단 · 안정화 확인 필수. 보유기간은 변동성 기반 반감기(5~20일)로 사이징.
        </p>
      </div>

      {/* Regime + summary */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="rounded px-3 py-2 border" style={{ borderColor: C.border }}>
          <div className="text-[11px]" style={{ color: C.gray }}>Regime Multiplier</div>
          <div className="text-[15px] font-bold" style={{ color: C.text }}>{mult.toFixed(2)} <span className="text-[11px]" style={{ color: C.gray }}>{regimeLabel}</span></div>
        </div>
        <div className="rounded px-3 py-2 border" style={{ borderColor: C.border }}>
          <div className="text-[11px]" style={{ color: C.gray }}>후보 수</div>
          <div className="text-[15px] font-bold" style={{ color: C.text }}>{data.length}</div>
        </div>
        <div className="rounded px-3 py-2 border" style={{ borderColor: C.border }}>
          <div className="text-[11px]" style={{ color: C.gray }}>평균 MR Score</div>
          <div className="text-[15px] font-bold" style={{ color: C.text }}>{avg.toFixed(1)}</div>
        </div>
        <div className="ml-auto flex rounded overflow-hidden border" style={{ borderColor: C.border }}>
          {(["ALL", "Stock", "ETF"] as const).map((t) => (
            <button key={t} onClick={() => setTypeFilter(t)}
              className="px-3 py-1 text-[12px] font-semibold"
              style={{ background: typeFilter === t ? C.purple : "transparent", color: typeFilter === t ? "#fff" : C.gray }}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {note && (
        <div className="text-[12px] rounded px-3 py-2" style={{ color: C.orange, background: C.orange + "12" }}>
          ⚠ {note}
        </div>
      )}

      {filtered.length === 0 ? (
        <div className="text-[13px] p-6 text-center rounded border" style={{ color: C.gray, borderColor: C.border }}>
          현재 Oversold Reversion 후보가 없습니다. {mult <= 0.5 && "강추세 레짐이라 MR이 억제된 상태입니다 (regime multiplier 낮음)."}
        </div>
      ) : (
        <div className="overflow-x-auto rounded border" style={{ borderColor: C.border }}>
          <table className="w-full text-[12px]">
            <thead className="sticky top-0" style={{ backgroundColor: C.bgAlt }}>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                <th className="text-left px-2 py-1.5" style={{ color: C.gray }}>Ticker</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }}>Type</th>
                <th className="text-left px-2 py-1.5" style={{ color: C.gray }}>분류</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray, minWidth: 70 }} title="regime-adjusted MR score">MR Score</th>
                <th className="text-left px-2 py-1.5" style={{ color: C.gray, minWidth: 160 }} title="5개 sub-signal 분해">Sub-signals (OU·Idio·Stab·LT·Stretch)</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="예상 반감기 (변동성 기반)">반감기</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }}>QVR</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }}>RSI</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="SMA20 이격도 (음수=평균 아래)">SMA20</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="52주 고점 대비">vs 52wHi</th>
                <th className="text-right px-2 py-1.5" style={{ color: C.gray }}>5d</th>
                <th className="text-right px-2 py-1.5" style={{ color: C.gray }}>1mo</th>
                <th className="text-right px-2 py-1.5" style={{ color: C.gray }}>3mo</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => (
                <tr key={d.ticker} style={{ borderBottom: `1px solid ${C.border}40` }}>
                  <td className="px-2 py-1.5">
                    <div className="font-mono font-bold" style={{ color: C.text }}>{d.ticker}</div>
                    <div className="text-[10px]" style={{ color: C.gray }}>{d.sector}</div>
                  </td>
                  <td className="text-center px-2 py-1.5">{d.asset_type === "ETF" ? "📦" : "📈"}</td>
                  <td className="px-2 py-1.5 text-[11px]" style={{ color: C.text }}>{d.classification}</td>
                  <td className="text-center px-2 py-1.5">
                    <span className="font-mono font-bold text-[14px]" style={{ color: C.purple }}>{(d.mr_score_adj ?? d.mr_score).toFixed(0)}</span>
                    {d.mr_score_adj != null && d.mr_score_adj !== d.mr_score && (
                      <div className="text-[9px]" style={{ color: C.gray }}>raw {d.mr_score.toFixed(0)}</div>
                    )}
                  </td>
                  <td className="px-2 py-1.5" style={{ minWidth: 160 }}>
                    <SubBar label="OU" v={d.mr_ou} color={C.cyan} />
                    <SubBar label="Idio" v={d.mr_idio} color={C.purple} />
                    <SubBar label="Stab" v={d.mr_stab} color={C.green} />
                    <SubBar label="LT" v={d.mr_lt} color={C.orange} />
                    <SubBar label="Strch" v={d.mr_stretch} color={C.gray} />
                  </td>
                  <td className="text-center px-2 py-1.5 font-mono" style={{ color: C.text }}>{d.mr_half_life_days}d</td>
                  <td className="text-center px-2 py-1.5 font-mono" style={{ color: d.qvr_score >= 55 ? C.green : d.qvr_score >= 45 ? C.text : C.orange }}>{d.qvr_score?.toFixed(0) ?? "—"}</td>
                  <td className="text-center px-2 py-1.5 font-mono" style={{ color: d.rsi < 30 ? C.green : C.text }}>{d.rsi?.toFixed(0) ?? "—"}</td>
                  <td className="text-center px-2 py-1.5 font-mono" style={{ color: pctColor(d.sma20_dist) }}>{fpct(d.sma20_dist)}</td>
                  <td className="text-center px-2 py-1.5 font-mono" style={{ color: pctColor(d.pct_from_high) }}>{fpct(d.pct_from_high)}</td>
                  <td className="text-right px-2 py-1.5 font-mono" style={{ color: pctColor(d.ret_5d) }}>{fpct(d.ret_5d)}</td>
                  <td className="text-right px-2 py-1.5 font-mono" style={{ color: pctColor(d.ret_1m) }}>{fpct(d.ret_1m)}</td>
                  <td className="text-right px-2 py-1.5 font-mono" style={{ color: pctColor(d.ret_3m) }}>{fpct(d.ret_3m)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Methodology note */}
      <div className="text-[11px] rounded border p-3 leading-relaxed" style={{ borderColor: C.border, color: C.gray }}>
        <strong style={{ color: C.text }}>방법론:</strong> OU dislocation z-score(변동성 정규화 SMA 이격) ·
        Idiosyncratic residual(섹터 대비 혼자 빠진 정도) · Stabilization(하락 소진 timing) ·
        Long-term reversal(De Bondt-Thaler 3-5년 패자) · Mean-distance stretch(볼린저/RSI 과매도).
        모멘텀 Composite와 <strong style={{ color: C.text }}>상호배타</strong>(모멘텀 게이트 탈락 종목만) ·
        regime-gated · 반감기 기반 청산(OU z가 0 통과 or 1.5×반감기 시간손절).
      </div>
    </div>
  );
}
