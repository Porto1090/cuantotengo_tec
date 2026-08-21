import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Durante desarrollo, /api y /static se redirigen al backend FastAPI
// (evita configurar CORS en el navegador durante local dev).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/static": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
