import { useEffect, useRef, useState } from "react";

// Por ahora vamos a manejar el progreso de manera aproximada, ya que no tenemos un endpoint que nos diga el progreso real del procesamiento de la imagen.
// En un futuro estaría bien poder hacer una inferencia local y no depender de apis externas como la de openai, permitiendonos tener un progreso real y no aproximado.

const ESTIMATED_TOTAL_MS = 7000;

const STAGES = [
  { upTo: 12, label: "Subiendo imagen..." },
  { upTo: 30, label: "Preprocesando imagen..." },
  { upTo: 88, label: "Detectando productos..." },
  { upTo: 97, label: "Preparando resultados..." },
  { upTo: 100, label: "¡Completado!" },
];

function stageLabelFor(percent) {
  const stage = STAGES.find((s) => percent <= s.upTo);
  return stage ? stage.label : STAGES[STAGES.length - 1].label;
}

export default function ProgressBar({ active, done }) {
  const [percent, setPercent] = useState(0);
  const startRef = useRef(null);
  const frameRef = useRef(null);

  useEffect(() => {
    if (!active) {
      setPercent(0);
      startRef.current = null;
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      return;
    }

    startRef.current = performance.now();

    function tick(now) {
      const elapsed = now - startRef.current;
      const raw = 1 - Math.exp(-elapsed / (ESTIMATED_TOTAL_MS * 0.6));
      const capped = Math.min(raw * 100, 97);
      setPercent(capped);
      frameRef.current = requestAnimationFrame(tick);
    }

    frameRef.current = requestAnimationFrame(tick);
    return () => frameRef.current && cancelAnimationFrame(frameRef.current);
  }, [active]);

  useEffect(() => {
    if (done) {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      setPercent(100);
    }
  }, [done]);

  if (!active && !done) return null;

  return (
    <div className="flex flex-col gap-2 rounded-[14px] border border-[#2a2a2e] bg-[#17171a] p-[16px_18px]" role="status" aria-live="polite">
      <div className="relative h-[8px] overflow-hidden rounded-full bg-[#202024]">
        <div
          className="relative h-full overflow-hidden rounded-full bg-[#f59e0b] transition-[width] duration-200 ease-linear"
          style={{ width: `${percent}%` }}
        >
          <span className="absolute inset-0 w-[40%] bg-gradient-to-r from-transparent via-white/45 to-transparent animate-[glint_1.3s_ease-in-out_infinite]" />
        </div>
      </div>
      <div className="flex justify-between text-[13px] text-[#a3a3a8]">
        <span>{stageLabelFor(percent)}</span>
        <span className="font-semibold tabular-nums text-[#f59e0b]">{Math.round(percent)}%</span>
      </div>
    </div>
  );
}