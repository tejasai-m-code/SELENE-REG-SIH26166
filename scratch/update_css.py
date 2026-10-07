import os

filepath = "artifacts/selene-reg-x/src/index.css"

new_css = """@import 'tailwindcss';
@import 'tw-animate-css';
@plugin "@tailwindcss/typography";

@custom-variant dark (&:is(.dark *));

@theme inline {
  --color-background: hsl(var(--background));
  --color-foreground: hsl(var(--foreground));
  --color-border: hsl(var(--border));
  --color-input: hsl(var(--input));
  --color-ring: hsl(var(--ring));
  --color-card: hsl(var(--card));
  --color-card-foreground: hsl(var(--card-foreground));
  --color-card-border: hsl(var(--card-border));
  --color-primary: hsl(var(--primary));
  --color-primary-foreground: hsl(var(--primary-foreground));
  --color-secondary: hsl(var(--secondary));
  --color-secondary-foreground: hsl(var(--secondary-foreground));
  --color-muted: hsl(var(--muted));
  --color-muted-foreground: hsl(var(--muted-foreground));
  --color-accent: hsl(var(--accent));
  --color-accent-foreground: hsl(var(--accent-foreground));
  --color-destructive: hsl(var(--destructive));
  --color-destructive-foreground: hsl(var(--destructive-foreground));
  --font-sans: var(--app-font-sans);
  --font-mono: var(--app-font-mono);
  --font-display: var(--app-font-display);
  --radius-sm: calc(var(--radius) - 4px);
  --radius-md: calc(var(--radius) - 2px);
  --radius-lg: var(--radius);
}

:root {
  /* DARK SCIENTIFIC THEME AS DEFAULT */
  --background: 222 47% 4%;
  --foreground: 210 40% 98%;
  --border: 217 33% 17%;
  --input: 217 33% 17%;
  --ring: 204 94% 45%;
  
  --card: 222 47% 8%;
  --card-foreground: 210 40% 98%;
  --card-border: 217 33% 18%;
  
  --primary: 204 94% 45%;
  --primary-foreground: 0 0% 100%;
  
  --secondary: 217 33% 12%;
  --secondary-foreground: 210 40% 98%;
  
  --muted: 217 33% 12%;
  --muted-foreground: 215 20% 65%;
  
  --accent: 38 92% 50%;
  --accent-foreground: 0 0% 100%;
  
  --destructive: 0 72% 51%;
  --destructive-foreground: 210 40% 98%;

  --popover: 222 47% 8%;
  --popover-foreground: 210 40% 98%;
  
  --app-font-sans: 'Instrument Sans', 'Geist', 'Inter', -apple-system, sans-serif;
  --app-font-mono: 'DM Mono', 'JetBrains Mono', 'IBM Plex Mono', monospace;
  --app-font-display: 'Instrument Sans', 'Space Grotesk', sans-serif;
  --radius: 0.375rem;
  
  --shadow-xs: 0 1px 2px 0 rgba(0, 0, 0, 0.5);
  --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.6), 0 1px 2px -1px rgba(0, 0, 0, 0.5);
  --shadow-md: 0 4px 12px -2px rgba(0, 0, 0, 0.6), 0 2px 4px -1px rgba(0, 0, 0, 0.4);
}

@layer base {
  * { @apply border-border; }
  body { @apply font-sans antialiased bg-background text-foreground; }
  button, input, select { font: inherit; }
}

html { scroll-behavior: smooth; }
body { min-width: 320px; }

/* Micro-Grid & Laboratory Instrument Canvas */
.instrument-grid {
  background-color: transparent;
  background-image: 
    linear-gradient(to right, rgba(148, 163, 184, 0.05) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(148, 163, 184, 0.05) 1px, transparent 1px);
  background-size: 24px 24px;
}

.panel {
  background: hsl(var(--card));
  border: 1px solid hsl(var(--card-border));
  box-shadow: var(--shadow-sm);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.mono { font-family: var(--app-font-mono); }
.font-display { font-family: var(--app-font-display); }
.eyebrow {
  font-family: var(--app-font-mono);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.data-value {
  font-family: var(--app-font-mono);
  font-variant-numeric: tabular-nums;
}

.focus-ring:focus-visible {
  outline: 2px solid #0f4c81;
  outline-offset: 2px;
}

/* Scientific Accent Shimmers & Pulses */
@keyframes jewel-shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(200%); }
}

.animate-jewel-shimmer {
  position: relative;
  overflow: hidden;
}

.animate-jewel-shimmer::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
  animation: jewel-shimmer 3.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
}

@keyframes light-sweep {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.shimmer-badge-verified {
  background: linear-gradient(90deg, rgba(52, 211, 153, 0.1) 0%, rgba(52, 211, 153, 0.2) 35%, rgba(255, 255, 255, 0.2) 50%, rgba(52, 211, 153, 0.2) 65%, rgba(52, 211, 153, 0.1) 100%);
  background-size: 250% 100%;
  animation: light-sweep 4s ease-in-out infinite;
}

@keyframes vector-flow {
  to { stroke-dashoffset: -40; }
}

.vector-active-stream {
  stroke-dasharray: 6 4;
  animation: vector-flow 1.4s linear infinite;
}

@keyframes active-stepper-glow {
  0%, 100% { box-shadow: 0 0 0 1px rgba(15, 76, 129, 0.35), 0 1px 3px rgba(15, 76, 129, 0.15); }
  50% { box-shadow: 0 0 0 3px rgba(15, 76, 129, 0.2), 0 3px 8px rgba(15, 76, 129, 0.18); }
}

.stepper-glow-active {
  animation: active-stepper-glow 3s ease-in-out infinite;
}

.fade-up { animation: fade-up .4s cubic-bezier(0.16, 1, 0.3, 1) both; }
.delay-1 { animation-delay: 60ms; }
.delay-2 { animation-delay: 120ms; }
.delay-3 { animation-delay: 180ms; }

@keyframes fade-up {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    transition-duration: .01ms !important;
    scroll-behavior: auto !important;
  }
}

/* Subtle Technical Grid & Restrained Stars */
.stars {
  position: fixed; inset: 0; pointer-events: none; opacity: 0.15;
  background-image:
    radial-gradient(circle, #fff 1px, transparent 1px),
    radial-gradient(circle, #fff 1px, transparent 1px);
  background-size: 150px 150px, 200px 200px;
  background-position: 0 0, 50px 50px;
  z-index: -2;
}
.grid-bg {
  position: fixed; inset: 0; pointer-events: none; opacity: 0.03;
  background-image: 
    linear-gradient(#38bdf8 1px, transparent 1px),
    linear-gradient(90deg, #38bdf8 1px, transparent 1px);
  background-size: 50px 50px;
  z-index: -1;
}

/* Custom Form Controls */
select {
  -webkit-appearance: none;
  appearance: none;
  background-image: url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right 0.5rem center;
  background-size: 1em;
  padding-right: 2rem !important;
}

option {
  background-color: hsl(var(--card)) !important;
  color: hsl(var(--foreground)) !important;
}
"""

with open(filepath, "w", encoding="utf-8") as f:
    f.write(new_css)

print("Updated index.css")
