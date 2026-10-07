import re

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the uncalibrated badge in PairCompatibilityCard
content = content.replace(\"'bg-slate-200 text-slate-300'\", \"'bg-[#19BFF5]/10 text-[#19BFF5] border border-[#19BFF5]/30'\")
content = content.replace(\"compatLevel === 'uncalibrated' ? 'bg-cyan-50 text-[#8A97A8] border-cyan-200'\", \"compatLevel === 'uncalibrated' ? 'bg-[#19BFF5]/10 text-[#19BFF5] border border-[#19BFF5]/30'\")

with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed.')
