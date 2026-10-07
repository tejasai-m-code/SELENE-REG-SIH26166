import os
import re

directories = ["artifacts/selene-reg-x/src"]

replacements = {
    r'\bbg-white\b': 'bg-card',
    r'\bbg-slate-50\b': 'bg-muted',
    r'\bbg-slate-100\b': 'bg-muted',
    r'\bbg-slate-200\b': 'bg-accent',
    r'\btext-slate-900\b': 'text-card-foreground',
    r'\btext-slate-800\b': 'text-card-foreground',
    r'\btext-slate-700\b': 'text-muted-foreground',
    r'\btext-slate-600\b': 'text-muted-foreground',
    r'\btext-slate-500\b': 'text-muted-foreground',
    r'\bborder-slate-200\b': 'border-border',
    r'\bborder-slate-300\b': 'border-border',
    r'\bborder-slate-400\b': 'border-border',
    r'\bfrom-white\b': 'from-card',
    r'\bto-slate-50\b': 'to-muted',
    r'\bbg-cyan-50\b': 'bg-primary/10',
    r'\btext-cyan-800\b': 'text-primary',
    r'\btext-cyan-900\b': 'text-primary',
    r'\btext-cyan-700\b': 'text-primary',
    r'\bborder-cyan-300\b': 'border-primary/20',
    r'\bborder-cyan-200\b': 'border-primary/20',
    r'\bbg-emerald-50\b': 'bg-emerald-500/10',
    r'\btext-emerald-800\b': 'text-emerald-500',
    r'\btext-emerald-900\b': 'text-emerald-500',
    r'\btext-emerald-700\b': 'text-emerald-500',
    r'\bborder-emerald-300\b': 'border-emerald-500/20',
    r'\bbg-rose-50\b': 'bg-destructive/10',
    r'\btext-rose-800\b': 'text-destructive',
    r'\btext-rose-900\b': 'text-destructive',
    r'\bborder-rose-300\b': 'border-destructive/20',
    r'\bbg-amber-50\b': 'bg-amber-500/10',
    r'\btext-amber-800\b': 'text-amber-500',
    r'\btext-amber-900\b': 'text-amber-500',
    r'\bborder-amber-300\b': 'border-amber-500/20'
}

for d in directories:
    for root, _, files in os.walk(d):
        for filename in files:
            if filename.endswith(".tsx") or filename.endswith(".ts"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = content
                for pattern, replacement in replacements.items():
                    new_content = re.sub(pattern, replacement, new_content)
                
                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated {filepath}")

print("Done")
