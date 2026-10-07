with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('function Workstation()')
idx2 = content.find(' {/* Right Column: Preview & Diagnostics */}', idx)

workstation_panel = content[idx:idx2]

workstation_panel = workstation_panel.replace('className="panel fade-up overflow-hidden rounded-lg"', 'className="fade-up overflow-hidden rounded-xl bg-[#0B1119] border border-[#1A2638] shadow-[0_0_15px_rgba(25,191,245,0.03)]"')

workstation_panel = workstation_panel.replace('border-b border-slate-700/80 px-4 py-3 bg-slate-900', 'border-b border-[#1A2638] px-4 py-3 bg-[#0D1520]')
workstation_panel = workstation_panel.replace('eyebrow text-sky-800', 'eyebrow text-[#8A97A8]')

workstation_panel = workstation_panel.replace('bg-slate-800/80', 'bg-[#111B29]')
workstation_panel = workstation_panel.replace('bg-slate-800/60', 'bg-[#111B29]')
workstation_panel = workstation_panel.replace('bg-slate-900/50', 'bg-[#111B29]')
workstation_panel = workstation_panel.replace('bg-slate-800', 'bg-[#142033]')
workstation_panel = workstation_panel.replace('border-slate-700/50', 'border-[#1A2638]')

workstation_panel = workstation_panel.replace('text-slate-100', 'text-[#E8EEF5]')
workstation_panel = workstation_panel.replace('text-slate-200', 'text-[#E8EEF5]')
workstation_panel = workstation_panel.replace('text-slate-300', 'text-[#E8EEF5]')
workstation_panel = workstation_panel.replace('text-slate-400', 'text-[#8A97A8]')
workstation_panel = workstation_panel.replace('text-slate-500', 'text-[#8A97A8]')
workstation_panel = workstation_panel.replace('text-cyan-500', 'text-[#19BFF5]')
workstation_panel = workstation_panel.replace('border-cyan-800/50', 'border-[#19BFF5]/30')

workstation_panel = workstation_panel.replace('accent-amber-500', 'accent-[#F5A400]')
workstation_panel = workstation_panel.replace('border-amber-500/50', 'border-[#F5A400]')
workstation_panel = workstation_panel.replace('text-amber-500', 'text-[#F5A400]')
workstation_panel = workstation_panel.replace('bg-slate-700', 'bg-[#142033]')

old_run_btn = 'bg-sky-700 hover:bg-sky-800 text-sm font-mono font-bold tracking-wider text-white shadow-xs transition-all duration-150 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-40 border border-sky-800'
new_run_btn = 'bg-[#19BFF5] hover:bg-[#15A5D6] text-sm font-mono font-bold tracking-wider text-[#05080B] shadow-[0_0_15px_rgba(25,191,245,0.2)] transition-all duration-150 active:scale-[0.99] disabled:cursor-not-allowed disabled:opacity-40 border border-[#19BFF5]'
workstation_panel = workstation_panel.replace(old_run_btn, new_run_btn)

content = content[:idx] + workstation_panel + content[idx2:]

with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(content)
print('Panel updated.')
