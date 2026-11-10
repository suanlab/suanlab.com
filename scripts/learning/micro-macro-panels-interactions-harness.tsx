import React, { StrictMode, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { LanguageProvider, useLanguage } from '../../src/components/language-provider';
import { SiteMotionProvider, MotionControls } from '../../src/components/motion/motion-provider';
import MicroMacroLab from '../../src/components/learning/micro-macro-lab';
import { microMacroPresets, microMacroGuides, type MicroMacroKind } from '../../src/lib/learning-micro-macro';
import { microMacroInputKey } from '../../src/lib/learning-micro-macro-protocol';
const interactionProbe = { started: 0, acknowledged: 0, terminated: 0, live: 0, ticks: 0 };
Object.assign(window, { microMacroInteractionProbe: interactionProbe });
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
Object.assign(window, { microMacroPanelGuides: Object.fromEntries(Object.entries(microMacroGuides).map(([kind, guides]) => [kind, guides.map(g => g.preset)])), microMacroPanelPresets: Object.fromEntries(Object.entries(microMacroPresets).map(([kind, presets]) => [kind, presets.map(p => ({ label: p.label, key: microMacroInputKey(p.input) }))])) });
function Harness() {
  const [kind, setKind] = useState<MicroMacroKind>('binary-shell-multiplicity-block-counts'), { setLanguage } = useLanguage();
  return <div className="ai-book"><nav aria-label="Verification controls" style={{ padding: '1rem', display: 'flex', gap: '.5rem', flexWrap: 'wrap' }}>
    {Object.keys(microMacroPresets).map((k, i) => <button type="button" data-testid={`kind-${i}`} key={k} onClick={() => setKind(k as MicroMacroKind)}>Family {i + 1}</button>)}
    <button type="button" data-testid="language-ko" onClick={() => setLanguage('ko')}>KO</button><button type="button" data-testid="language-en" onClick={() => setLanguage('en')}>EN</button>
    <button type="button" data-testid="theme" onClick={() => document.documentElement.classList.toggle('dark')}>Theme</button><MotionControls />
    </nav><main id="main-content" style={{ width: 'min(100%, 800px)', padding: '.5rem', margin: 'auto' }}><MicroMacroLab kind={kind} /></main></div>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><LanguageProvider><SiteMotionProvider><Harness /></SiteMotionProvider></LanguageProvider></StrictMode>);
