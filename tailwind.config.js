/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
        "blob": {
          "0%": { transform: "translate(0px, 0px) scale(1)" },
          "33%": { transform: "translate(30px, -50px) scale(1.1)" },
          "66%": { transform: "translate(-20px, 20px) scale(0.9)" },
          "100%": { transform: "translate(0px, 0px) scale(1)" },
        },
        "spin-slow": {
          from: { transform: "rotate(0deg)" },
          to: { transform: "rotate(360deg)" },
        },
        "float": {
          "0%, 100%": { transform: "translateY(0) rotate(45deg)" },
          "50%": { transform: "translateY(-20px) rotate(45deg)" },
        },
        "gradient-x": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        "fade-in-up": {
          "0%": { opacity: 0, transform: "translateY(20px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
        "data-stream": {
          "0%": { opacity: 0, transform: "translateY(-100%)" },
          "50%": { opacity: 1 },
          "100%": { opacity: 0, transform: "translateY(100%)" },
        },
        "stream-down": {
          "0%": { transform: "translateY(-100%)", opacity: 0 },
          "10%": { opacity: 0.8 },
          "90%": { opacity: 0.8 },
          "100%": { transform: "translateY(500%)", opacity: 0 },
        },
        "particle-1": {
          "0%, 100%": { transform: "translate(0, 0)", opacity: 0.4 },
          "25%": { transform: "translate(15px, -10px)", opacity: 0.5 },
          "50%": { transform: "translate(30px, 5px)", opacity: 0.3 },
          "75%": { transform: "translate(15px, 15px)", opacity: 0.45 },
        },
        "particle-2": {
          "0%, 100%": { transform: "translate(0, 0)", opacity: 0.35 },
          "33%": { transform: "translate(-12px, -18px)", opacity: 0.5 },
          "66%": { transform: "translate(10px, -10px)", opacity: 0.25 },
        },
        "particle-3": {
          "0%, 100%": { transform: "translate(0, 0)", opacity: 0.4 },
          "50%": { transform: "translate(-25px, 12px)", opacity: 0.55 },
        },
        "particle-4": {
          "0%, 100%": { transform: "translate(0, 0)", opacity: 0.3 },
          "20%": { transform: "translate(18px, 12px)", opacity: 0.4 },
          "40%": { transform: "translate(10px, -15px)", opacity: 0.5 },
          "60%": { transform: "translate(-10px, -6px)", opacity: 0.35 },
          "80%": { transform: "translate(-15px, 10px)", opacity: 0.45 },
        },
        "node-pulse": {
          "0%, 100%": { opacity: 0.85 },
          "50%": { opacity: 1 },
        },
        "ripple": {
          "0%": { transform: "scale(1)", opacity: 0.4 },
          "100%": { transform: "scale(1.8)", opacity: 0 },
        },
        "signal-flow": {
          "0%": { strokeDashoffset: 110 },
          "100%": { strokeDashoffset: 0 },
        },
        "signal-flow-long": {
          "0%": { strokeDashoffset: 210 },
          "100%": { strokeDashoffset: 0 },
        },
        "scan-line": {
          "0%": { transform: "translateY(0)" },
          "100%": { transform: "translateY(400px)" },
        },
        "core-pulse": {
          "0%, 100%": { opacity: 0.15 },
          "50%": { opacity: 0.25 },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "blob": "blob 7s infinite",
        "spin-slow": "spin-slow 20s linear infinite",
        "float": "float 6s ease-in-out infinite",
        "gradient-x": "gradient-x 3s ease infinite",
        "fade-in-up": "fade-in-up 0.6s ease-out forwards",
        "data-stream": "data-stream 4s ease-in-out infinite",
        "stream-down": "stream-down 5s ease-in-out infinite",
        "particle-1": "particle-1 8s ease-in-out infinite",
        "particle-2": "particle-2 10s ease-in-out infinite",
        "particle-3": "particle-3 7s ease-in-out infinite",
        "particle-4": "particle-4 12s ease-in-out infinite",
        "node-pulse": "node-pulse 2s ease-in-out infinite",
        "ripple": "ripple 2s ease-out infinite",
        "signal-flow": "signal-flow 1.5s ease-in-out infinite",
        "signal-flow-long": "signal-flow-long 2.5s ease-in-out infinite",
        "scan-line": "scan-line 4s linear infinite",
        "core-pulse": "core-pulse 4s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
