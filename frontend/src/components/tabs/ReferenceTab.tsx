import { useEffect, useState, useMemo } from "react";
import axios from "axios";
import { C } from "../../styles/theme";

// ───── Types ─────

interface RefItem {
  id: string;
  filename: string;
  title: string;
  authors: string;
  year: string;
  venue: string;
  applies_to: string[];
  category: string;
  available: boolean;
  size_kb: number;
}

interface CitationOnlyItem {
  title: string;
  authors: string;
  year: string;
  venue: string;
  url?: string;
  applies_to: string;
  note?: string;
}

interface ReferenceData {
  downloaded: RefItem[];
  citation_only: CitationOnlyItem[];
  bibliography: { filename: string; available: boolean };
}

// ───── Category grouping ─────

const CATEGORY_ORDER = [
  "Momentum (Foundational)",
  "Momentum (Multi-Asset)",
  "Momentum (Cross-Asset)",
  "Momentum (Robustness)",
  "Momentum (52-Week High)",
  "Momentum (Risk Management)",
  "Behavioral (Underreaction)",
  "Behavioral (Information Diffusion)",
  "Behavioral (Overconfidence)",
  "Behavioral (Overreaction)",
  "Factor Model (Foundational)",
  "Factor Model (Methodology)",
  "Factor Model (Momentum Factor)",
  "Factor Model (Quality)",
  "Factor Model (BAB)",
  "Technical Analysis (Foundational)",
  "Technical Analysis (MA Rules)",
  "Volatility (Foundational)",
  "Risk-Adjusted Performance",
  "Option Theory (Reference)",
];

// Group categories into super-categories for display
const SUPER_CATEGORIES: { label: string; color: string; categories: string[] }[] = [
  {
    label: "Momentum",
    color: "#0A7D3F",
    categories: [
      "Momentum (Foundational)", "Momentum (Multi-Asset)", "Momentum (Cross-Asset)",
      "Momentum (Robustness)", "Momentum (52-Week High)", "Momentum (Risk Management)",
    ],
  },
  {
    label: "Behavioral Finance",
    color: "#7D5BA6",
    categories: [
      "Behavioral (Underreaction)", "Behavioral (Information Diffusion)",
      "Behavioral (Overconfidence)", "Behavioral (Overreaction)",
    ],
  },
  {
    label: "Factor Models",
    color: "#0F5499",
    categories: [
      "Factor Model (Foundational)", "Factor Model (Methodology)",
      "Factor Model (Momentum Factor)", "Factor Model (Quality)", "Factor Model (BAB)",
    ],
  },
  {
    label: "Technical Analysis",
    color: "#B85C00",
    categories: ["Technical Analysis (Foundational)", "Technical Analysis (MA Rules)"],
  },
  {
    label: "Volatility & Risk",
    color: "#CC0000",
    categories: ["Volatility (Foundational)", "Risk-Adjusted Performance", "Option Theory (Reference)"],
  },
];

// ───── Helper ─────

const fetchReferences = () => axios.get("/api/references").then((r) => r.data);

function pdfUrl(filename: string): string {
  return `/api/references/file/${encodeURIComponent(filename)}`;
}

// ───── Components ─────

function RefCard({ ref }: { ref: RefItem }) {
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3 hover:border-[#E6D9CE] transition-colors">
      <div className="flex items-start justify-between gap-3 mb-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-2 mb-1">
            <span className="text-[12px] font-mono px-1.5 py-0.5 rounded bg-[#F2E5D7]/60 text-[#0F5499] shrink-0">
              [{ref.id}]
            </span>
            <span className="text-[14px] font-bold text-[#0D7680] leading-snug">
              {ref.title}
            </span>
          </div>
          <div className="text-[13px] text-[#66605C] italic">
            {ref.authors} ({ref.year})
          </div>
          <div className="text-[12px] text-[#857F7A] mt-0.5">{ref.venue}</div>
        </div>
        <div className="shrink-0 flex flex-col items-end gap-1">
          {ref.available ? (
            <a
              href={pdfUrl(ref.filename)}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[12px] px-2 py-1 rounded bg-[#E3EEF5]/40 text-[#0D7680] hover:bg-[#E3EEF5]/60 font-semibold whitespace-nowrap"
              title={`${ref.size_kb} KB`}
            >
              📄 View PDF
            </a>
          ) : (
            <span className="text-[12px] px-2 py-1 rounded bg-[#F2E5D7] text-[#857F7A]">unavailable</span>
          )}
          <a
            href={pdfUrl(ref.filename)}
            download={ref.filename}
            className="text-[11px] text-[#857F7A] hover:text-[#33302E]"
          >
            ↓ download
          </a>
        </div>
      </div>

      {ref.applies_to.length > 0 && (
        <div className="mt-2 pt-2 border-t border-[#E6D9CE]/60">
          <div className="text-[12px] text-[#857F7A] font-semibold mb-1 uppercase tracking-wide">
            Applies to
          </div>
          <ul className="space-y-0.5">
            {ref.applies_to.map((item, i) => (
              <li key={i} className="text-[10.5px] text-[#66605C] flex gap-1.5">
                <span className="text-[#857F7A] shrink-0">•</span>
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function CitationOnlyCard({ item }: { item: CitationOnlyItem }) {
  return (
    <div className="bg-[#FBEEE3] border border-[#E6D9CE]/60 rounded-lg p-3 border-dashed">
      <div className="text-[14px] font-bold text-[#33302E] leading-snug mb-1">
        {item.title}
      </div>
      <div className="text-[13px] text-[#857F7A] italic mb-1">
        {item.authors} ({item.year})
      </div>
      <div className="text-[12px] text-[#857F7A] mb-2">{item.venue}</div>
      {item.url && item.url !== "—" && (
        <div className="text-[10.5px] text-[#0F5499] break-all mb-2">
          <a href={item.url} target="_blank" rel="noopener noreferrer" className="hover:underline">
            🔗 {item.url}
          </a>
        </div>
      )}
      <div className="text-[10.5px] text-[#66605C]">
        <span className="text-[#857F7A] font-semibold">Applies: </span>
        {item.applies_to}
      </div>
      {item.note && (
        <div className="text-[12px] text-[#C2701C]/70 italic mt-1.5">
          ⚠ {item.note}
        </div>
      )}
    </div>
  );
}

// ───── Main Tab ─────

export function ReferenceTab() {
  const [data, setData] = useState<ReferenceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  const [expandedSuper, setExpandedSuper] = useState<Set<string>>(new Set(SUPER_CATEGORIES.map((s) => s.label)));

  useEffect(() => {
    setLoading(true);
    fetchReferences().then(setData).finally(() => setLoading(false));
  }, []);

  // Group refs by category
  const byCategory = useMemo(() => {
    if (!data) return {};
    const map: Record<string, RefItem[]> = {};
    for (const r of data.downloaded) {
      map[r.category] = map[r.category] || [];
      map[r.category].push(r);
    }
    return map;
  }, [data]);

  // Filter logic
  const filterFn = (r: RefItem) => {
    if (!filter.trim()) return true;
    const q = filter.toLowerCase();
    return (
      r.title.toLowerCase().includes(q) ||
      r.authors.toLowerCase().includes(q) ||
      r.year.includes(q) ||
      r.venue.toLowerCase().includes(q) ||
      r.applies_to.some((a) => a.toLowerCase().includes(q))
    );
  };

  const toggleSuper = (label: string) => {
    setExpandedSuper((s) => {
      const newSet = new Set(s);
      if (newSet.has(label)) newSet.delete(label);
      else newSet.add(label);
      return newSet;
    });
  };

  if (loading) return <div className="text-[#857F7A] p-8">Loading references...</div>;
  if (!data) return <div className="text-[#857F7A] p-8">No data</div>;
  if ("error" in (data as any)) {
    return <div className="text-[#CC0000] p-8">Error: {(data as any).error}</div>;
  }

  const totalDownloaded = data.downloaded.length;
  const totalAvailable = data.downloaded.filter((r) => r.available).length;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-[20px] font-bold text-[#33302E]">Academic References</h2>
        <p className="text-[14px] text-[#857F7A] mt-1">
          Price Discovery 시스템에 사용된 모든 정량 로직의 학술적 근거.
          총 {totalDownloaded}개 논문 다운로드 ({totalAvailable}개 사용 가능) + {data.citation_only.length}개 citation-only.
        </p>
      </div>

      {/* Top actions */}
      <div className="flex items-center gap-3 flex-wrap">
        {data.bibliography.available && (
          <a
            href={pdfUrl(data.bibliography.filename)}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[14px] px-3 py-2 rounded bg-[#E3EEF5] hover:bg-[#0F5499] text-white font-semibold"
          >
            📚 Bibliography 종합 PDF 보기
          </a>
        )}

        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="🔍 검색 (제목/저자/연도/적용처...)"
          className="flex-1 min-w-[280px] px-3 py-2 text-[14px] bg-[#F2E5D7] border border-[#E6D9CE] rounded text-[#33302E] placeholder-gray-500 focus:outline-none focus:border-[#9CC3D5]"
        />

        <button
          onClick={() => setExpandedSuper(new Set(SUPER_CATEGORIES.map((s) => s.label)))}
          className="text-[12px] px-2 py-1 rounded border border-[#E6D9CE] text-[#857F7A] hover:text-[#33302E]"
        >
          Expand All
        </button>
        <button
          onClick={() => setExpandedSuper(new Set())}
          className="text-[12px] px-2 py-1 rounded border border-[#E6D9CE] text-[#857F7A] hover:text-[#33302E]"
        >
          Collapse All
        </button>
      </div>

      {/* Super-categories */}
      {SUPER_CATEGORIES.map((sc) => {
        // Collect all refs in this super-category
        const refs: RefItem[] = [];
        for (const cat of sc.categories) {
          for (const r of byCategory[cat] || []) {
            if (filterFn(r)) refs.push(r);
          }
        }
        if (refs.length === 0 && filter) return null; // hide empty when filtering

        const isOpen = expandedSuper.has(sc.label);

        return (
          <div key={sc.label} className="border border-[#E6D9CE] rounded-lg overflow-hidden">
            <button
              onClick={() => toggleSuper(sc.label)}
              className="w-full px-4 py-3 text-left bg-[#FFFFFF] hover:bg-[#F2E5D7] flex items-center justify-between"
              style={{ borderLeft: `3px solid ${sc.color}` }}
            >
              <div className="flex items-center gap-3">
                <span className="text-[16px] font-bold" style={{ color: sc.color }}>
                  {sc.label}
                </span>
                <span className="text-[12px] text-[#857F7A]">
                  {refs.length} {refs.length === 1 ? "paper" : "papers"}
                </span>
              </div>
              <span className="text-[#857F7A] text-[14px]">{isOpen ? "▼" : "▶"}</span>
            </button>

            {isOpen && (
              <div className="bg-[#FBEEE3] p-3 space-y-3">
                {sc.categories.map((cat) => {
                  const catRefs = (byCategory[cat] || []).filter(filterFn);
                  if (catRefs.length === 0) return null;
                  return (
                    <div key={cat}>
                      <div className="text-[12px] font-semibold text-[#857F7A] uppercase tracking-wider mb-2 px-1">
                        {cat}
                      </div>
                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
                        {catRefs.map((r) => (
                          <RefCard key={r.id} ref={r} />
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}

      {/* Citation-only section */}
      {data.citation_only.length > 0 && (
        <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
          <div className="px-4 py-3 bg-[#FFFFFF]" style={{ borderLeft: "3px solid #999" }}>
            <span className="text-[16px] font-bold text-[#66605C]">Citation-Only References</span>
            <span className="text-[12px] text-[#857F7A] ml-2">
              {data.citation_only.length} (paywall, blocked, or book)
            </span>
          </div>
          <div className="bg-[#FBEEE3] p-3 grid grid-cols-1 xl:grid-cols-2 gap-2">
            {data.citation_only.map((c, i) => (
              <CitationOnlyCard key={i} item={c} />
            ))}
          </div>
        </div>
      )}

      {/* Footer info */}
      <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3">
        <p className="text-[13px] text-[#857F7A]">
          파일 위치: <code className="text-[12px] bg-[#F2E5D7] px-1 py-0.5 rounded text-[#66605C]">
            docs/references/
          </code>
          &nbsp;&nbsp;|&nbsp;&nbsp; PDF 형식 표준 학술 인용 가능. 다운로드 후 Endnote/Mendeley/Zotero 등으로 import.
        </p>
      </div>
    </div>
  );
}
