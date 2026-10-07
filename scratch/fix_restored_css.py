import re

filepath = "artifacts/selene-reg-x/src/index.css"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix font variables
content = content.replace("var(--app-font-primary)", "var(--app-font-display)")

# Add missing global typography if not present
typography_css = """
h1, h2, h3, h4 {
    color: hsl(var(--foreground));
    font-family: var(--app-font-display);
    letter-spacing: -0.02em;
}
"""

if "h1, h2, h3, h4" not in content:
    content += "\n" + typography_css

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed restored CSS.")
