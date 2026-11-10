import React, { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LanguageProvider, useLanguage } from '../../src/components/language-provider';
import { SiteMotionProvider, MotionControls } from '../../src/components/motion/motion-provider';
import EnsembleLab from '../../src/components/learning/ensemble-lab';
import { ensemblePresets, ensembleGuides, type EnsembleKind } from '../../src/lib/learning-ensembles';
import { ensembleInputKey } from '../../src/lib/learning-ensemble-protocol';
const interactionProbe = { started: 0, acknowledged: 0, terminated: 0, live: 0, ticks: 0 };
Object.assign(window, { ensembleInteractionProbe: interactionProbe });
const NativeWorker = window.Worker;
class TrackedWorker extends NativeWorker {
  private stopped = false;
  constructor(url: string | URL, options?: WorkerOptions) {
    super(url, options); interactionProbe.started++; interactionProbe.live++;
    this.addEventListener('message', event => { if (event.data?.phase === 'started') interactionProbe.acknowledged++; });
  }
  terminate() { if (!this.stopped) { this.stopped = true; interactionProbe.terminated++; interactionProbe.live--; } super.terminate(); }
}
window.Worker = TrackedWorker;
setInterval(() => interactionProbe.ticks++, 16);
Object.assign(window, { ensemblePanelGuides: Object.fromEntries(Object.entries(ensembleGuides).map(([kind, guides]) => [kind, guides.map(g => g.preset)])), ensemblePanelPresets: Object.fromEntries(Object.entries(ensemblePresets).map(([kind, presets]) => [kind, presets.map(p => ({ label: p.label, key: ensembleInputKey(p.input) }))])) });
function Harness() {
  const [kind, setKind] = useState<EnsembleKind>('finite-binary-reservoir-canonical-comparison'), { setLanguage } = useLanguage();
  return <div className="ai-book"><nav aria-label="Verification controls" style={{ padding: '1rem', display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
    {Object.keys(ensemblePresets).map((k, i) => <button type="button" data-testid={`kind-${i}`} key={k} onClick={() => setKind(k as EnsembleKind)}>Family {i + 1}</button>)}
    <button type="button" data-testid="language-ko" onClick={() => setLanguage('ko')}>KO</button><button type="button" data-testid="language-en" onClick={() => setLanguage('en')}>EN</button>
    <button type="button" data-testid="theme" onClick={() => document.documentElement.classList.toggle('dark')}>Theme</button><MotionControls />
    </nav><main id="main-content" style={{ width: 'min(100%, 800px)', padding: '.5rem', margin: 'auto' }}><EnsembleLab kind={kind} /></main></div>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><LanguageProvider><SiteMotionProvider><Harness /></SiteMotionProvider></LanguageProvider></StrictMode>);
