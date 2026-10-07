css_fixes = """
/* Missing utilities for injected header/hero */
.bg-panel2 { background-color: hsl(var(--card)); }
.border-line { border-color: hsl(var(--border)); }
.bg-warning { background-color: var(--warning); }
.bg-error { background-color: hsl(var(--destructive)); }
.bg-success { background-color: var(--success); }
.text-text { color: hsl(var(--foreground)); }
.text-muted { color: hsl(var(--muted-foreground)); }
.hover\\:text-accent:hover { color: hsl(var(--primary)); }
"""

filepath = "artifacts/selene-reg-x/src/index.css"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

if "Missing utilities for injected header" not in content:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n" + css_fixes)
    print("Added missing utilities to index.css")
else:
    print("Already added.")
