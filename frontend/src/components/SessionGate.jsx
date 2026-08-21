import { useState } from "react";
import { enterSession, enterTestSession } from "../api.js";

export default function SessionGate({ onEnter }) {
  const [sessionInput, setSessionInput] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleEnter() {
    setError(null);
    setLoading(true);
    try {
      const data = await enterSession(sessionInput);
      if (!data.ok) {
        setError(data.error || "ID inválido (1-999)");
        return;
      }
      onEnter(data.session_id, data.enter_time);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleTestEnter() {
    setError(null);
    setLoading(true);
    try {
      const data = await enterTestSession();
      onEnter(data.session_id, data.enter_time);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") handleEnter();
  }

  return (
    <section className="flex justify-center pt-8">
      <div className="flex w-full max-w-[380px] flex-col gap-[14px] rounded-[20px] border border-[#2a2a2e] bg-[#17171a] p-[28px_24px] shadow-[0_8px_24px_rgba(0,0,0,0.35)]">
        <h2 className="m-0 font-display text-[22px] font-semibold text-[#f5f5f5]">
          Ingresa tu ID de usuario
        </h2>
        <p className="-mt-2 mb-1 text-[13.5px] text-[#a3a3a8]">
          Usa el ID de 1 a 3 dígitos asignado a tu sesión.
        </p>

        <label className="m-0 text-[12.5px] uppercase tracking-[0.04em] text-[#a3a3a8]" htmlFor="session-input">
          ID de sesión
        </label>
        <input
          id="session-input"
          className="w-full rounded-[8px] border border-[#2a2a2e] bg-[#0b0b0c] px-[14px] py-[13px] text-[16px] text-[#f5f5f5] transition-colors duration-150 focus-visible:border-[#f59e0b] focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-[rgba(245,158,11,0.14)]"
          placeholder="EJEMPLO: 000"
          value={sessionInput}
          onChange={(e) => setSessionInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          autoFocus
        />

        {error && <p className="m-0 text-[14px] font-medium text-[#f87171]">{error}</p>}

        <div className="mt-[6px] flex flex-col gap-[10px]">
          <button
            className="w-full rounded-[8px] border-none bg-[#f59e0b] px-5 py-[14px] text-[15px] font-semibold text-[#1a1200] transition-all duration-150 hover:brightness-[1.08] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-55 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f59e0b]"
            onClick={handleEnter}
            disabled={loading}
          >
            {loading ? "Ingresando..." : "ENTRAR"}
          </button>
          <button
            className="w-full rounded-[8px] border border-[#f59e0b] bg-transparent px-5 py-[14px] text-[15px] font-semibold text-[#f59e0b] transition-all duration-150 hover:bg-[rgba(245,158,11,0.14)] active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-55 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#f59e0b]"
            onClick={handleTestEnter}
            disabled={loading}
          >
            ENTRAR COMO TEST
          </button>
        </div>
      </div>
    </section>
  );
}