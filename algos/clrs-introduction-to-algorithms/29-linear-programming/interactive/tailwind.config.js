/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,svelte}",
  ],
  theme: {
    extend: {},
  },
  plugins: [require('daisyui')],
  daisyui: {
    themes: [
      {
        lpvisualizer: {
          "primary": "#a78bfa",
          "secondary": "#c084fc",
          "accent": "#f9a8d4",
          "neutral": "#374151",
          "base-100": "#ffffff",
          "base-200": "#f3f4f6",
          "base-300": "#e5e7eb",
          "info": "#93c5fd",
          "success": "#86efac",
          "warning": "#fcd34d",
          "error": "#fca5a5",
        },
      },
    ],
  },
}
