import React, { StrictMode, useEffect, useLayoutEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { useEnsembleCalculation } from '../../src/components/learning/use-ensemble-calculation';
import { ensembleInputKey, type EnsembleInput } from '../../src/lib/learning-ensemble-protocol';
import { ensemblePresetInput } from '../../src/lib/learning-ensembles';
const reservoir = ensemblePresetInput('finite-binary-reservoir-canonical-comparison', 0);
const canonical = ensemblePresetInput('canonical-binary-energy-fluctuations', 0);
const grand = ensemblePresetInput('grand-canonical-finite-occupancy-fluctuations', 4);
const heavy: EnsembleInput = { kind: 'finite-binary-reservoir-canonical-comparison', parameters: { subsystemSites: 2048, reservoirSites: 2048, totalExcitations: 1 } };
const probe = { started: 0, acknowledged: 0, terminated: 0, live: 0, heartbeats: 0, maxGap: 0, clicks: 0, staleResults: 0, renders: 0,
  latest: { status: '', phase: null as string | null, kind: '', size: 0, requestId: null as number | null, value: null as number | null } };
Object.assign(window, { ensembleProbe: probe });
const NativeWorker = window.Worker;
class TrackedWorker extends NativeWorker {
  private finished = false;
  constructor(url: string | URL, options?: WorkerOptions) {
    super(url, options); probe.started++; probe.live++;
    this.addEventListener('message', event => { if (event.data?.phase === 'started') probe.acknowledged++; });
  }
  terminate() { if (!this.finished) { this.finished = true; probe.terminated++; probe.live--; } super.terminate(); }
}
window.Worker = TrackedWorker;
let last = performance.now();
setInterval(() => { const now = performance.now(); probe.heartbeats++; probe.maxGap = Math.max(probe.maxGap, now - last); last = now; }, 16);
function Lab({ input, enabled }: { input: EnsembleInput; enabled: boolean }) {
  // Recreating the object must not restart work on ordinary React renders.
  const result = useEnsembleCalculation(structuredClone(input), enabled);
  useLayoutEffect(() => {
    probe.renders++;
    const response = result.response;
    if (response && (response.kind !== input.kind || !Object.entries(input.parameters).every(([key, value]) => (response.result.parameters as unknown as Record<string, unknown>)[key] === value))) probe.staleResults++;
    probe.latest = { status: result.status, phase: result.phase, kind: input.kind,
      size: input.kind === 'finite-binary-reservoir-canonical-comparison' ? input.parameters.subsystemSites : input.parameters.sites,
      requestId: result.requestId, value: response?.kind === 'finite-binary-reservoir-canonical-comparison' ? response.result.microToCanonicalKlNats.value
        : response?.kind === 'canonical-binary-energy-fluctuations' ? response.result.heatCapacityInKB.value : response?.kind === 'grand-canonical-finite-occupancy-fluctuations' ? response.result.numberVariance.value : null };
  });
  return <section data-input-key={ensembleInputKey(input)}><p data-testid="status">{result.status}</p>
    <button data-testid="cancel" onClick={result.cancel}>Cancel</button><button data-testid="retry" onClick={result.retry}>Retry</button></section>;
}
function Harness() {
  const [input, setInput] = useState<EnsembleInput>(reservoir), [mounted, setMounted] = useState(true), [clicks, setClicks] = useState(0), [enabled, setEnabled] = useState(true);
  useEffect(() => { probe.clicks = clicks; }, [clicks]);
  return <main><button data-testid="reservoir" onClick={() => setInput(reservoir)}>Reservoir</button><button data-testid="canonical" onClick={() => setInput(canonical)}>Canonical</button>
    <button data-testid="grand" onClick={() => setInput(grand)}>Grand</button><button data-testid="heavy" onClick={() => setInput(heavy)}>Heavy</button>
    <button data-testid="toggle" onClick={() => setMounted(value => !value)}>Toggle lab</button><button data-testid="click" onClick={() => setClicks(value => value + 1)}>Respond {clicks}</button>
    <button data-testid="enabled" onClick={() => setEnabled(value => !value)}>Enable calculation</button>{mounted && <Lab input={input} enabled={enabled} />}</main>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><Harness /></StrictMode>);
