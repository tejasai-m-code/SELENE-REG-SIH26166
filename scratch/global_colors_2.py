import re

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all bg-cyan-100, bg-sky-50, bg-emerald-50, bg-orange-50, bg-amber-50 with their dark equivalents
content = re.sub(r'bg-cyan-100(/\d+)?', 'bg-[#19BFF5]/10', content)
content = re.sub(r'bg-sky-50(/\d+)?', 'bg-[#19BFF5]/10', content)
content = re.sub(r'bg-emerald-50(/\d+)?', 'bg-emerald-500/10', content)
content = re.sub(r'bg-orange-50(/\d+)?', 'bg-[#F5A400]/10', content)
content = re.sub(r'bg-amber-50(/\d+)?', 'bg-[#F5A400]/10', content)

# Replace all text-cyan-800, text-sky-800, text-sky-900, text-emerald-800, text-orange-900 with their dark theme equivalents
content = re.sub(r'text-cyan-800', 'text-[#19BFF5]', content)
content = re.sub(r'text-cyan-900', 'text-[#19BFF5]', content)
content = re.sub(r'text-sky-800', 'text-[#19BFF5]', content)
content = re.sub(r'text-sky-900', 'text-[#19BFF5]', content)
content = re.sub(r'text-emerald-800', 'text-emerald-400', content)
content = re.sub(r'text-orange-900', 'text-[#F5A400]', content)
content = re.sub(r'text-amber-800', 'text-[#F5A400]', content)
content = re.sub(r'text-red-900', 'text-red-400', content)
content = re.sub(r'text-red-800', 'text-red-400', content)

# Replace border-cyan-200, border-sky-200, border-emerald-200, border-orange-200
content = re.sub(r'border-cyan-200', 'border-[#19BFF5]/30', content)
content = re.sub(r'border-cyan-300', 'border-[#19BFF5]/30', content)
content = re.sub(r'border-cyan-400', 'border-[#19BFF5]/50', content)
content = re.sub(r'border-sky-200', 'border-[#19BFF5]/30', content)
content = re.sub(r'border-sky-300', 'border-[#19BFF5]/30', content)
content = re.sub(r'border-emerald-200', 'border-emerald-500/30', content)
content = re.sub(r'border-emerald-300', 'border-emerald-500/30', content)
content = re.sub(r'border-orange-200', 'border-[#F5A400]/30', content)
content = re.sub(r'border-amber-200', 'border-[#F5A400]/30', content)

with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('Global replacements done phase 2.')
