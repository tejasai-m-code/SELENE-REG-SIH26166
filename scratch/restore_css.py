css_to_append = """
/* ============================================================
   PROTOTYPE 2 UI COMPONENTS (RESTORED HEADER & HERO)
   ============================================================ */

/* Navigation */
.topbar {
    position: sticky; top: 0; z-index: 100;
    height: 64px; padding: 0 2rem;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    background: rgba(2, 2, 2, 0.8); backdrop-filter: blur(12px);
    font-family: var(--app-font-mono);
    font-size: 13px;
}
.brand { display: flex; gap: 1rem; align-items: center; }
.brand-name { font-family: var(--app-font-primary); font-weight: 700; letter-spacing: 0.1em; font-size: 16px; color: var(--foreground); }
.brand-name span { color: hsl(var(--primary)); }
.brand-sub { color: hsl(var(--muted-foreground)); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; }

.nav-links { display: flex; gap: 2rem; }
.nav-links a { color: hsl(var(--muted-foreground)); text-decoration: none; transition: color 0.2s; text-transform: uppercase; letter-spacing: 0.1em; font-family: var(--app-font-mono); font-size: 11px; }
.nav-links a:hover, .nav-links a.active { color: var(--foreground); text-shadow: 0 0 10px rgba(56, 189, 248, 0.3); }
.pill { font-family: var(--app-font-mono); font-size: 10px; font-weight: 500; background: rgba(56, 189, 248, 0.12); color: hsl(var(--primary)); padding: 3px 10px; border-radius: 2px; letter-spacing: 0.05em; }
.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: hsl(var(--success, 150 100% 50%)); margin-right: 6px; box-shadow: 0 0 6px rgba(52, 211, 153, 0.5); }

/* Hero Section */
.hero-container {
    display: grid;
    grid-template-columns: 1fr 400px 1fr;
    align-items: center;
    gap: 2rem;
    min-height: calc(100vh - 64px);
    padding: 2rem;
    max-width: 1600px;
    margin: 0 auto;
}

.hero-left h1 { font-size: 3rem; font-weight: 500; line-height: 1.1; margin: 0 0 1rem; letter-spacing: -0.02em; font-family: var(--app-font-display); color: var(--foreground); }
.hero-left h1 em { font-style: normal; color: hsl(var(--primary)); }
.hero-left p { color: hsl(var(--muted-foreground)); font-size: 1.1rem; line-height: 1.6; max-width: 400px; font-family: var(--app-font-sans); }

.hero-center { display: grid; place-items: center; position: relative; }
.hero-right { display: flex; flex-direction: column; gap: 1rem; }
.hero-right .eyebrow { font-family: var(--app-font-mono); color: hsl(var(--primary)); font-size: 12px; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0; }
.hero-right p { color: hsl(var(--muted-foreground)); font-size: 1rem; line-height: 1.6; max-width: 350px; margin: 0; font-family: var(--app-font-sans); }

/* Moon Video */
.moon-video {
    width: 100%; max-width: 380px; aspect-ratio: 1;
    object-fit: cover;
    border-radius: 50%;
    pointer-events: none;
    -webkit-mask-image: radial-gradient(circle, black 60%, transparent 71%);
    mask-image: radial-gradient(circle, black 60%, transparent 71%);
    filter: drop-shadow(0 0 40px rgba(56, 189, 248, 0.08));
}
"""

filepath = "artifacts/selene-reg-x/src/index.css"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

if "PROTOTYPE 2 UI COMPONENTS (RESTORED HEADER & HERO)" not in content:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + css_to_append)
    print("Appended UI Components to index.css")
else:
    print("Already appended.")
