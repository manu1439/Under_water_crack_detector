/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#070b14',
          900: '#0b1329',
          800: '#111e3f',
          700: '#1a2b58',
          600: '#263d7a',
        },
        ocean: {
          500: '#0ea5e9',
          400: '#38bdf8',
          600: '#0284c7',
        }
      }
    },
  },
  plugins: [],
}
