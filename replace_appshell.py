import re

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read()

# Normalize line endings
content = content.replace('\r\n', '\n')

start_idx = content.find('  return (\n    <div className="min-h-[100dvh] bg-[#f8fafc] text-slate-900 font-sans">')

end_str = '      <main className="bg-[#f8fafc]">{children}</main>\n    </div>\n  );'
end_idx = content.find(end_str, start_idx)

if start_idx != -1 and end_idx != -1:
    end_idx += len(end_str)
    
    replacement = '''  return (
    <div className="min-h-[100dvh] flex flex-col relative">
      <div className="stars"></div>
      <div className="grid-bg"></div>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">◒</div>
          <div>
            <div className="brand-name">SELENE<span>-REG</span></div>
            <div className="brand-sub">LUNAR IMAGE CORRESPONDENCE ENGINE</div>
          </div>
        </div>
        <div className="nav-links hidden lg:flex">
          {nav.map(([key, href, label, Icon]) => (
            <Link
              key={key}
              href={href}
              className={current === key ? 'active' : ''}
            >
              {label}
            </Link>
          ))}
        </div>
        <div className="header-right hidden md:flex items-center gap-2.5">
          <span className="pill">SIH 26166</span>
          <HardwareStatusBar
            requestedMode={requestedMode}
            onSelectMode={onSelectMode}
            runtimeTelemetry={runtimeTelemetry}
            isProcessing={isProcessing}
          />
        </div>
      </header>
      <nav aria-label="Primary navigation" className="nav-links flex border-b border-line bg-panel2 px-4 py-3 lg:hidden overflow-x-auto gap-4">
        {nav.map(([key, href, label, Icon]) => (
          <Link
            key={key}
            href={href}
            className={`flex items-center gap-1 shrink-0 ${current === key ? 'active' : ''}`}
          >
            {label}
          </Link>
        ))}
      </nav>
      {children}
    </div>
  );'''
    
    new_content = content[:start_idx] + replacement + content[end_idx:]
    with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Replaced AppShell successfully.")
else:
    print(f"Could not find start or end indices. start={start_idx}, end={end_idx}")
