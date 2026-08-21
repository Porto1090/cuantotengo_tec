/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          DEFAULT: "#0b0b0c",
          grid: "#131315",
        },
        surface: {
          DEFAULT: "#17171a",
          2: "#202024",
        },
        border: {
          DEFAULT: "#2a2a2e",
        },
        accent: {
          DEFAULT: "#f59e0b",
          dim: "rgba(245, 158, 11, 0.14)",
        },
        text: {
          DEFAULT: "#f5f5f5",
          dim: "#a3a3a8",
        },
        error: {
          DEFAULT: "#f87171",
          bg: "rgba(248, 113, 113, 0.1)",
        },
        row: {
          even: "#1c1c1f",
          odd: "#262629",
        },
      },
      fontFamily: {
        display: ["Space Grotesk", "Inter", "system-ui", "sans-serif"],
        body: ["Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        sm: "8px",
        DEFAULT: "14px",
        lg: "20px",
      },
      boxShadow: {
        card: "0 8px 24px rgba(0, 0, 0, 0.35)",
      },
      keyframes: {
        scan: {
          "0%": { top: "8%", opacity: "0" },
          "15%": { opacity: "0.7" },
          "50%": { top: "92%", opacity: "0.7" },
          "65%": { opacity: "0" },
          "100%": { top: "92%", opacity: "0" },
        },
        glint: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(320%)" },
        },
      },
      animation: {
        scan: "scan 3.2s ease-in-out infinite",
        glint: "glint 1.3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};