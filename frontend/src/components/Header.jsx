export default function Header() {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[#2a2a2e] pb-[18px]">
      <div className="flex items-center gap-3">
        <span 
          className="grid h-10 w-10 flex-shrink-0 place-items-center rounded-[8px] bg-[#f59e0b] font-display text-[15px] font-bold text-[#1a1200]" 
          aria-hidden="true"
        >
          CT
        </span>
        <div>
          <h1 className="m-0 font-display text-[26px] font-bold tracking-[-0.01em] text-[#f5f5f5] min-[480px]:text-[32px]">
            CuantoTengo
          </h1>
          <p className="mb-0 mt-[2px] text-[13px] text-[#a3a3a8]">
            Conteo de productos por visión artificial
          </p>
        </div>
      </div>
      <span className="whitespace-nowrap text-[12px] font-semibold tracking-[0.08em] text-[#a3a3a8]">
        BY LIFT LAB
      </span>
    </header>
  );
}