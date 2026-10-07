import re

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace green subtitles/labels (sky-800, cyan-800, cyan-900) in eyebrow/text
# We want these to be near-white or muted cool gray. Let's make them muted cool gray (#8A97A8)
content = content.replace('text-sky-800', 'text-[#8A97A8]')
content = content.replace('text-cyan-800', 'text-[#8A97A8]')
content = content.replace('text-cyan-900', 'text-[#E8EEF5]')

# Replace white background cards (bg-cyan-50) with dark graphite (#111B29)
content = re.sub(r'bg-cyan-50(/\d+)?', 'bg-[#111B29]', content)

# Replace other white icons/bubbles
content = content.replace('bg-white', 'bg-[#142033]')

# Fix specific button in Validation page (OpenCV baseline)
# The button in Validation is rendered with:
# method === item ? 'border-cyan-700 bg-cyan-50 font-semibold text-cyan-900'
content = content.replace('border-cyan-700 bg-[#111B29] font-semibold text-[#E8EEF5]', 'border-[#19BFF5] bg-[#111B29] font-semibold text-[#19BFF5] shadow-[0_0_10px_rgba(25,191,245,0.15)]')

# Fix large circular icon above "Awaiting Image Strip"
# originally: 'flex size-14 items-center justify-center rounded-full bg-cyan-50 border border-cyan-200'
content = content.replace('bg-[#111B29] border border-cyan-200', 'bg-[#0B1119] border border-[#19BFF5]/30 shadow-[0_0_15px_rgba(25,191,245,0.15)]')

# Also fix the icon color inside it:
content = content.replace('<Layers3 className="size-7 text-cyan-700" />', '<Layers3 className="size-7 text-[#19BFF5]" />')

# Let's fix Raster Manifest & Compatibility Card (PairCompatibilityCard)
# The "UNCALIBRATED" badge
content = content.replace('bg-orange-50 text-orange-900 border-orange-200', 'bg-[#F5A400]/10 text-[#F5A400] border-[#F5A400]/30')
content = content.replace('bg-emerald-50 text-emerald-900 border-emerald-200', 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30')
content = content.replace('bg-cyan-50 text-[#8A97A8] border-cyan-200', 'bg-[#19BFF5]/10 text-[#19BFF5] border-[#19BFF5]/30')

with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)

print('Global replacements done.')
