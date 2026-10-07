import re

def fix_hardware_status_bar():
    filepath = "artifacts/selene-reg-x/src/components/HardwareStatusBar.tsx"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # CPU active class
    content = content.replace(
        "bg-accent text-foreground font-bold border border-border shadow-xs",
        "bg-card text-foreground font-bold border border-border shadow-xs"
    )

    # GPU active class
    content = content.replace(
        "bg-accent text-emerald-500 font-bold border border-emerald-500/50 shadow-xs",
        "bg-primary/10 text-primary font-bold border border-primary/40 shadow-xs"
    )
    # GPU dot
    content = content.replace(
        "requestedMode === 'gpu' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'",
        "requestedMode === 'gpu' ? 'bg-primary animate-pulse shadow-[0_0_8px_rgba(56,189,248,0.6)]' : 'bg-muted-foreground'"
    )

    # HYBRID active class
    content = content.replace(
        "bg-accent text-primary font-bold border border-sky-500/50 shadow-xs",
        "bg-primary/10 text-primary font-bold border border-primary/40 shadow-xs"
    )
    # HYBRID dot
    content = content.replace(
        "requestedMode === 'hybrid' ? 'bg-[#0f4c81] animate-pulse' : 'bg-slate-400'",
        "requestedMode === 'hybrid' ? 'bg-primary animate-pulse shadow-[0_0_8px_rgba(56,189,248,0.6)]' : 'bg-muted-foreground'"
    )
    
    # Inactive dots
    content = content.replace("'bg-slate-400'", "'bg-muted-foreground'")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed HardwareStatusBar.tsx")

def fix_appshell():
    filepath = "artifacts/selene-reg-x/src/App.tsx"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    start_idx = content.find("function AppShell({")
    if start_idx == -1:
        print("AppShell not found in App.tsx")
        return

    end_str = "    </div>\n  );\n}"
    end_idx = content.find(end_str, start_idx)
    if end_idx == -1:
        print("AppShell end not found")
        return
    end_idx += len(end_str)

    new_appshell = """function AppShell({
  children,
  current,
  requestedMode,
  onSelectMode,
  runtimeTelemetry,
  isProcessing,
}: {
  children: ReactNode;
  current: string;
  requestedMode?: 'cpu' | 'gpu' | 'hybrid';
  onSelectMode?: (mode: 'cpu' | 'gpu' | 'hybrid') => void;
  runtimeTelemetry?: RuntimeTelemetry;
  isProcessing?: boolean;
}) {
  const nav = [
    ['workstation', '/', 'Workstation', Crosshair, 'Core ingestion & matching'],
    ['mosaic', '/mosaic', 'Multi-Image Mosaic', Layers3, 'Global graph registration'],
    ['audit', '/audit', 'Audit / PS-26166', ClipboardCheck, 'ISRO pipeline guidelines'],
    ['validation', '/validation', 'Validation', BarChart3, 'Runtime & hardware analytics']
  ] as const;

  return (
    <div className="min-h-[100dvh] flex flex-col relative">
      <div className="stars"></div>
      <div className="grid-bg"></div>
      
      {/* LAYER 1: BRAND + SYSTEM STATUS */}
      <header className="topbar !border-b-0">
        <div className="brand">
          <div className="brand-mark">◒</div>
          <div>
            <div className="brand-name">SELENE<span>-REG</span></div>
            <div className="brand-sub">LUNAR IMAGE CORRESPONDENCE ENGINE</div>
          </div>
        </div>
        <div className="header-right hidden md:flex items-center gap-4">
          <span className="pill">SIH 26166</span>
          <HardwareStatusBar
            requestedMode={requestedMode}
            onSelectMode={onSelectMode}
            runtimeTelemetry={runtimeTelemetry}
            isProcessing={isProcessing}
          />
        </div>
      </header>

      {/* LAYER 2: PRIMARY SYSTEM NAVIGATION */}
      <nav aria-label="Primary navigation" className="border-b border-border bg-background/50 backdrop-blur-sm z-10 w-full px-4 lg:px-8 py-3 mb-6">
        <div className="max-w-[1600px] mx-auto w-full">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {nav.map(([key, href, label, Icon, description]) => {
              const isActive = current === key;
              return (
                <Link
                  key={key}
                  href={href}
                  className={`group flex flex-col justify-center gap-1.5 p-3 rounded-md border transition-all ${
                    isActive 
                      ? 'border-primary/40 bg-primary/5 shadow-[0_0_15px_rgba(56,189,248,0.05)]' 
                      : 'border-border/60 bg-card/40 hover:bg-card hover:border-border'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Icon className={`size-4 transition-colors ${isActive ? 'text-primary' : 'text-muted-foreground group-hover:text-foreground'}`} />
                    <span className={`font-mono font-semibold text-[11px] tracking-[0.08em] uppercase transition-colors ${isActive ? 'text-primary' : 'text-foreground'}`}>
                      {label}
                    </span>
                  </div>
                  <span className={`text-[10px] leading-relaxed transition-colors ${isActive ? 'text-primary/70' : 'text-muted-foreground group-hover:text-muted-foreground/80'}`}>
                    {description}
                  </span>
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* LAYER 3: PAGE CONTENT */}
      <div className="flex-1 w-full">
        {children}
      </div>
    </div>
  );
}"""

    new_content = content[:start_idx] + new_appshell + content[end_idx:]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Fixed AppShell in App.tsx")

if __name__ == "__main__":
    fix_hardware_status_bar()
    fix_appshell()
