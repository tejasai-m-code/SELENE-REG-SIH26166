import os
import re

directories = ["artifacts/selene-reg-x/src"]

replacements = {
    r'\bbg-slate-950\b': 'bg-background',
    r'\bbg-slate-900\b': 'bg-card',
    r'\bbg-slate-800\b': 'bg-muted',
    r'\bbg-slate-700\b': 'bg-accent',
    
    r'\btext-slate-400\b': 'text-muted-foreground',
    r'\btext-slate-300\b': 'text-muted-foreground',
    r'\btext-slate-200\b': 'text-foreground',
    r'\btext-slate-100\b': 'text-foreground',
    
    r'\bborder-slate-800\b': 'border-border',
    r'\bborder-slate-700\b': 'border-border',
    r'\bborder-slate-600\b': 'border-border',
    r'\bborder-slate-500\b': 'border-border',
    
    r'\bbg-cyan-900\b': 'bg-primary/20',
    r'\bbg-cyan-800\b': 'bg-primary/20',
    r'\btext-cyan-400\b': 'text-primary',
    r'\btext-cyan-500\b': 'text-primary',
    r'\bborder-cyan-900\b': 'border-primary/20',
    r'\bborder-cyan-800\b': 'border-primary/20',
    
    r'\bbg-sky-900\b': 'bg-primary/20',
    r'\bbg-sky-800\b': 'bg-primary/20',
    r'\btext-sky-400\b': 'text-primary',
    r'\btext-sky-500\b': 'text-primary',
    
    r'\bbg-emerald-900\b': 'bg-emerald-500/20',
    r'\bbg-emerald-800\b': 'bg-emerald-500/20',
    r'\btext-emerald-400\b': 'text-emerald-500',
    
    r'\bbg-amber-900\b': 'bg-amber-500/20',
    r'\bbg-amber-800\b': 'bg-amber-500/20',
    r'\btext-amber-400\b': 'text-amber-500',
    
    r'\bbg-rose-900\b': 'bg-destructive/20',
    r'\bbg-rose-800\b': 'bg-destructive/20',
    r'\btext-rose-400\b': 'text-destructive',
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
