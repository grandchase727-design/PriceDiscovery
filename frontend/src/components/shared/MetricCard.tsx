export function MetricCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-[#FFFFFF] rounded-lg p-4 border border-[#E6D9CE]">
      <div className="text-[14px] text-[#857F7A] uppercase tracking-wide">{label}</div>
      <div className="text-[26px] font-bold text-[#33302E] mt-1">{value}</div>
      {sub && <div className="text-[14px] text-[#857F7A] mt-1">{sub}</div>}
    </div>
  );
}
