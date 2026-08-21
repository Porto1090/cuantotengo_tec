// Todas las llamadas al backend FastAPI viven aquí.
// En dev, Vite las redirige a http://localhost:8000 (ver vite.config.js).
// En producción, define VITE_API_BASE_URL apuntando a la URL real del backend.

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

async function parseJsonOrThrow(res) {
  let data;
  try {
    data = await res.json();
  } catch {
    throw new Error("Respuesta inválida del servidor.");
  }
  if (!res.ok) {
    throw new Error(data?.error || "Error de comunicación con el servidor.");
  }
  return data;
}

/** Equivalente a set_session_id() para un ID ingresado manualmente. */
export async function enterSession(sessionId) {
  const res = await fetch(`${BASE_URL}/api/session/enter`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return parseJsonOrThrow(res);
}

/** Equivalente al botón "ENTRAR COMO TEST". */
export async function enterTestSession() {
  const res = await fetch(`${BASE_URL}/api/session/test`, { method: "POST" });
  return parseJsonOrThrow(res);
}

/** Equivalente a process(): sube la imagen y devuelve los resultados de detección. */
export async function processImage(file, sessionId, enterTime) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("session_id", sessionId);
  if (enterTime) formData.append("enter_time", enterTime);

  const res = await fetch(`${BASE_URL}/api/process`, {
    method: "POST",
    body: formData,
  });
  return parseJsonOrThrow(res);
}

export function resolveImageUrl(url) {
  if (!url) return null;
  return url.startsWith("http") ? url : `${BASE_URL}${url}`;
}
