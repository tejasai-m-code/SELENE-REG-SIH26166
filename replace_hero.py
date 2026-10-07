import sys

with open('artifacts/selene-reg-x/src/App.tsx', 'r', encoding='utf-8') as f:
    content = f.read().replace('\r\n', '\n')

ws_start = content.find('function Workstation()')

# Find the start of the block to replace
start_str = '        <div className="mb-5 flex flex-col justify-between gap-3 lg:flex-row lg:items-end">'
start_idx = content.find(start_str, ws_start)

# Find the end of the block to replace
end_str = '        <div className="mb-4">\n          <WorkflowBreadcrumb currentStage={currentWorkflowStage} />\n        </div>'
end_idx = content.find(end_str, ws_start)

if start_idx == -1 or end_idx == -1:
    print('Block not found')
    sys.exit(1)

end_idx += len(end_str)

replacement = '''        <section className="hero-container mb-8">
          <div className="hero-left">
            <p className="eyebrow">ISRO SPACE TECHNOLOGY</p>
            <h1>Multi-modal lunar image<br/><em>correspondence & registration.</em></h1>
            <p>A resilient coarse-to-fine pipeline for Chandrayaan-2 optical imagery featuring scale, illumination, and cross-representation invariance.</p>
          </div>
          <div className="hero-center">
            <video className="moon-video" src="/assets/hero/selene-moon-loop.mp4" autoPlay muted playsInline loop preload="auto"></video>
          </div>
          <div className="hero-right">
            <p className="eyebrow">SCIENTIFIC WORKSTATION</p>
            <p>Black space. Lunar data. Measurable evidence. Proceed below to ingest scientific datasets and compute subpixel consensus registration.</p>
            <div className="nav-links mt-4" style={{flexDirection: 'column', gap: '0.5rem', fontFamily: 'var(--font-mono)'}}>
              <a href="#pairWorkspace">01. PAIR REGISTRATION</a>
              <a href="/mosaic">02. MULTI-IMAGE GRAPH</a>
              <a href="/validation">03. DIAGNOSTICS & RUNTIME</a>
            </div>
            
            <div className="mt-6 fade-up delay-1 flex items-center gap-2 self-start rounded-md border border-line bg-panel2 px-3 py-1.5 shadow-2xs font-mono">
              <span className={`size-2 rounded-full ${health.isLoading ? 'bg-warning animate-pulse' : health.isError ? 'bg-error' : 'bg-success animate-pulse'}`} />
              <span className="mono text-[11px] font-bold uppercase tracking-wider text-text">
                {health.isLoading ? 'checking API' : health.isError ? 'API unavailable' : `API ${health.data?.status || 'online'}`}
              </span>
              <button type="button" onClick={() => health.refetch()} className="focus-ring rounded p-0.5 text-muted hover:text-accent transition">
                <RefreshCw className="size-3" />
              </button>
            </div>
          </div>
        </section>
        
        <section className="workflow-tracker mb-8">
          <div className={`workflow-step ${currentWorkflowStage === 'dataset' ? 'active' : ''}`}><span className="step-no">01</span><span className="step-name">INGEST</span><span className="step-status">READY</span></div>
          <div className={`workflow-step ${['preprocess', 'detection', 'matching'].includes(currentWorkflowStage) ? 'active' : ''}`}><span className="step-no">02</span><span className="step-name">CORRESPOND</span><span className="step-status">WAITING</span></div>
          <div className={`workflow-step ${['geometry', 'subpixel', 'registration'].includes(currentWorkflowStage) ? 'active' : ''}`}><span className="step-no">03</span><span className="step-name">REGISTER</span><span className="step-status">WAITING</span></div>
          <div className={`workflow-step ${currentWorkflowStage === 'validation' ? 'active' : ''}`}><span className="step-no">04</span><span className="step-name">EVALUATE</span><span className="step-status">WAITING</span></div>
        </section>'''

new_content = content[:start_idx] + replacement + content[end_idx:]

with open('artifacts/selene-reg-x/src/App.tsx', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Replaced Header Block with Hero Section successfully.')
