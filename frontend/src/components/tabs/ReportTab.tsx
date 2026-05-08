import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchReport, type FilterParams } from "../../api/client";

export function ReportTab({ filters }: { filters: FilterParams }) {
  const [data, setData] = useState<any>(null);
  const [mode, setMode] = useState<"report" | "llm">("report");
  useEffect(() => { fetchReport(filters).then(setData); }, [filters]);
  if (!data) return <div className="text-gray-500 p-8">Loading...</div>;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <button
          className={`px-4 py-2 rounded text-sm ${mode === "report" ? "bg-blue-600 text-white" : "bg-[#1f2937] text-gray-400"}`}
          onClick={() => setMode("report")}>
          Auto Report
        </button>
        <button
          className={`px-4 py-2 rounded text-sm ${mode === "llm" ? "bg-blue-600 text-white" : "bg-[#1f2937] text-gray-400"}`}
          onClick={() => setMode("llm")}>
          LLM Prompt
        </button>
        <button
          className="px-3 py-2 text-xs bg-[#1f2937] border border-gray-700 rounded hover:bg-[#374151]"
          onClick={() => {
            const text = mode === "report" ? data.report_md : data.llm_prompt;
            const blob = new Blob([text], { type: "text/plain" });
            const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
            a.download = mode === "report" ? "report.md" : "llm_prompt.txt"; a.click();
          }}>
          Download
        </button>
      </div>

      {mode === "report" ? (
        <div className="bg-[#111827] border border-gray-800 rounded p-6 prose prose-invert prose-sm max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.report_md}</ReactMarkdown>
        </div>
      ) : (
        <div className="relative">
          <button
            className="absolute right-2 top-2 px-2 py-1 text-xs bg-[#374151] rounded hover:bg-[#4b5563]"
            onClick={() => navigator.clipboard.writeText(data.llm_prompt)}>
            Copy
          </button>
          <pre className="bg-[#111827] border border-gray-800 rounded p-4 text-xs text-gray-400 overflow-auto max-h-[80vh] whitespace-pre-wrap">
            {data.llm_prompt}
          </pre>
          <div className="text-xs text-gray-500 mt-2">{data.llm_prompt.length.toLocaleString()} chars</div>
        </div>
      )}
    </div>
  );
}
