import { useRef, useState } from "react";
import { processImage, resolveImageUrl } from "../api.js";
import ResultsTable from "./ResultsTable.jsx";
import ProgressBar from "./ProgressBar.jsx";

export default function Dashboard({ sessionId, enterTime }) {
  const fileInputRef = useRef(null);
  const [rows, setRows] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  async function runUpload(file) {
    if (!file) return;

    setErrorMessage("");
    setRows(null);
    setImageUrl(null);
    setDone(false);
    setHasResult(true);
    setLoading(true);

    try {
      const data = await processImage(file, sessionId, enterTime);

      if (!data.ok) {
        setErrorMessage(data.error || "No se pudo procesar la imagen. Vuelve a intentarlo.");
        return;
      }

      setRows(data.rows);
      setImageUrl(resolveImageUrl(data.image_url));
      setProcessingTime(data.processing_time);
    } catch (err) {
      setErrorMessage(err.message || "No se pudo procesar la imagen. Vuelve a intentarlo.");
    } finally {
      setDone(true);
      setTimeout(() => setLoading(false), 400);
    }
  }

  function handleFileChange(e) {
    runUpload(e.target.files?.[0]);
  }

  function handleDrop(e) {
    e.preventDefault();
    setIsDragging(false);
    if (hasResult) return;
    const file = e.dataTransfer.files?.[0];
    runUpload(file);
  }

  function handleReset() {
    setRows(null);
    setImageUrl(null);
    setErrorMessage("");
    setHasResult(false);
    setDone(false);
    setLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  const showUploadZone = !hasResult;
  const showWorkspace = hasResult;

  return (
    <section className="flex flex-col gap-[18px]">
      <div className="inline-flex self-start items-center gap-2 rounded-full border border-[#2a2a2e] bg-[#17171a] px-[14px] py-[7px] text-[13.5px] text-[#a3a3a8]">
        <span className="h-[7px] w-[7px] rounded-full bg-[#34d399] shadow-[0_0_0_3px_rgba(52,211,153,0.18)]" />
        Sesión <strong className="text-[#f5f5f5]">{sessionId === "000" ? "TEST" : sessionId}</strong>
      </div>

      {showUploadZone && (
        <label
          className={`relative flex flex-col items-center justify-center gap-2 overflow-hidden rounded-[20px] border-[1.5px] border-dashed px-[20px] py-[56px] text-center text-[#f5f5f5] cursor-pointer transition-colors duration-200 ${
            isDragging
              ? "border-[#f59e0b] bg-[rgba(245,158,11,0.14)]"
              : "border-[#2a2a2e] bg-[#17171a] hover:border-[#f59e0b] hover:bg-[rgba(245,158,11,0.14)]"
          }`}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          <div className="absolute left-0 right-0 h-[1.5px] bg-gradient-to-r from-transparent via-[#f59e0b] to-transparent opacity-60 animate-[scan_3.2s_ease-in-out_infinite] motion-reduce:hidden" />
          <svg className="mb-1 text-[#f59e0b]" width="40" height="40" viewBox="0 0 24 24" fill="none">
            <path
              d="M12 16V4M12 4L7 9M12 4l5 5M5 20h14"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="font-display text-[17px] font-semibold">Toma o sube una foto</span>
          <span className="text-[13px] text-[#a3a3a8]">Arrastra una imagen aquí o pulsa para elegirla</span>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>
      )}

      {loading && <ProgressBar active={loading} done={done} />}

      {errorMessage && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-[14px] border border-[rgba(248,113,113,0.3)] bg-[rgba(248,113,113,0.1)] p-[14px_16px]">
          <p className="m-0 text-[14px] font-medium text-[#f87171]">{errorMessage}</p>
          <button
            className="w-auto rounded-[8px] border border-[#f59e0b] bg-transparent px-4 py-[9px] text-[13.5px] font-semibold text-[#f59e0b] transition-all duration-150 hover:bg-[rgba(245,158,11,0.14)] active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f59e0b]"
            onClick={handleReset}
          >
            Reintentar
          </button>
        </div>
      )}

      {showWorkspace && (rows || imageUrl) && (
        <div className="flex flex-col gap-4 min-[860px]:flex-row min-[860px]:items-start">
          {imageUrl && (
            <div className="rounded-[20px] border border-[#2a2a2e] bg-[#17171a] p-[16px] shadow-[0_8px_24px_rgba(0,0,0,0.35)] min-[860px]:sticky min-[860px]:top-6 min-[860px]:flex-[0_0_42%]">
              <p className="m-0 mb-[10px] font-display text-[14px] font-semibold uppercase tracking-[0.04em] text-[#a3a3a8]">
                Imagen generada
              </p>
              <img className="block w-full rounded-[8px]" src={imageUrl} alt="Resultado anotado" />
            </div>
          )}

          {rows && (
            <div className="min-w-0 flex-1 rounded-[20px] border border-[#2a2a2e] bg-[#17171a] p-[16px] shadow-[0_8px_24px_rgba(0,0,0,0.35)]">
              <div className="mb-[10px] flex items-center justify-between">
                <p className="m-0 font-display text-[14px] font-semibold uppercase tracking-[0.04em] text-[#a3a3a8]">
                  Productos detectados
                </p>
                {processingTime && (
                  <span className="rounded-full bg-[rgba(245,158,11,0.14)] px-[9px] py-[3px] text-[12.5px] tabular-nums text-[#f59e0b]">
                    {processingTime.toFixed(2)}s
                  </span>
                )}
              </div>
              <ResultsTable rows={rows} />
            </div>
          )}
        </div>
      )}

      {hasResult && !loading && (
        <button
          className="w-full rounded-[8px] border-none bg-[#f59e0b] px-5 py-[14px] text-[15px] font-semibold text-[#1a1200] transition-all duration-150 hover:brightness-[1.08] active:scale-[0.98] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f59e0b]"
          onClick={handleReset}
        >
          TOMAR OTRA FOTO
        </button>
      )}
    </section>
  );
}