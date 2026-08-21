export default function ResultsTable({ rows }) {
  if (!rows || rows.length === 0) return null;

  return (
    <table className="w-full border-collapse overflow-hidden rounded-[8px] font-sans">
      <thead>
        <tr className="bg-[#202024] text-[#a3a3a8]">
          <th className="w-[56px] px-3 py-[10px] text-center text-[11.5px] font-semibold uppercase tracking-[0.05em]"></th>
          <th className="px-3 py-[10px] text-left text-[11.5px] font-semibold uppercase tracking-[0.05em]">MARCA</th>
          <th className="px-3 py-[10px] text-center text-[11.5px] font-semibold uppercase tracking-[0.05em]">TOTAL</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, idx) => (
          <tr 
            key={row.key} 
            className={idx % 2 === 0 ? "bg-[#1c1c1f]" : "bg-[#262629]"}
          >
            <td className="w-[56px] border-b border-[#2a2a2e] px-3 py-[10px] text-center last:border-b-0">
              <img
                src={row.image_url}
                alt={row.marca}
                className="h-10 w-10 rounded-[6px] bg-white/5 object-contain"
                onError={(e) => {
                  e.currentTarget.src = "https://via.placeholder.com/64";
                }}
              />
            </td>
            <td className="border-b border-[#2a2a2e] px-3 py-[10px] text-[15px] text-[#f5f5f5] last:border-b-0">
              {row.marca}
            </td>
            <td className="border-b border-[#2a2a2e] px-3 py-[10px] text-center text-[15px] font-bold tabular-nums text-[#f59e0b] last:border-b-0">
              {row.total}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}