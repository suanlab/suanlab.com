import React, { StrictMode, useEffect, useLayoutEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { usePartitionCalculation } from '../../src/components/learning/use-partition-calculation';
import { partitionInputKey, type PartitionInput } from '../../src/lib/learning-partition-protocol';
import { partitionPresetInput } from '../../src/lib/learning-partition';
const reservoir = partitionPresetInput('finite-spectrum-partition-cumulant-generators', 0);
const canonical = partitionPresetInput('coupled-binary-sites-partition-factorization', 0);
const grand = partitionPresetInput('infinite-energy-ladder-normalization-cutoff', 6);
const heavy: PartitionInput = { kind: 'infinite-energy-ladder-normalization-cutoff', parameters: { gapEnergy: 1.380649e-23 * .7, temperature: 1, cutoff: 4096 } };
const probe = { started: 0, acknowledged: 0, terminated: 0, live: 0, heartbeats: 0, maxGap: 0, clicks: 0, staleResults: 0, renders: 0,
  latest: { status: '', phase: null as string | null, kind: '', size: 0, requestId: null as number | null, value: null as number | null } };
Object.assign(window, { partitionProbe: probe });
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
function Lab({ input, enabled }: { input: PartitionInput; enabled: boolean }) {
  // Recreating the object must not restart work on ordinary React renders.
  const result = usePartitionCalculation(structuredClone(input), enabled);
  useLayoutEffect(() => {
    probe.renders++;
    const response = result.response;
    if (response && partitionInputKey({ kind: response.kind, parameters: response.result.parameters } as PartitionInput) !== partitionInputKey(input)) probe.staleResults++;
    probe.latest = { status: result.status, phase: result.phase, kind: input.kind,
      size: input.kind === 'finite-spectrum-partition-cumulant-generators' ? input.parameters.energyLevels.length : input.kind === 'coupled-binary-sites-partition-factorization' ? 4 : input.parameters.cutoff,
      requestId: result.requestId, value: response?.kind === 'finite-spectrum-partition-cumulant-generators' ? response.result.entropyInKB.value
        : response?.kind === 'coupled-binary-sites-partition-factorization' ? response.result.mutualInformation.value : response?.kind === 'infinite-energy-ladder-normalization-cutoff' ? response.result.finite.heatCapacityInKB.value : null };
  });
  return <section data-input-key={partitionInputKey(input)}><p data-testid="status">{result.status}</p>
    <button data-testid="cancel" onClick={result.cancel}>Cancel</button><button data-testid="retry" onClick={result.retry}>Retry</button></section>;
}
function Harness() {
  const [input, setInput] = useState<PartitionInput>(reservoir), [mounted, setMounted] = useState(true), [clicks, setClicks] = useState(0), [enabled, setEnabled] = useState(true);
  useEffect(() => { probe.clicks = clicks; }, [clicks]);
  return <main><button data-testid="reservoir" onClick={() => setInput(reservoir)}>Reservoir</button><button data-testid="canonical" onClick={() => setInput(canonical)}>Canonical</button>
    <button data-testid="grand" onClick={() => setInput(grand)}>Grand</button><button data-testid="heavy" onClick={() => setInput(heavy)}>Heavy</button>
    <button data-testid="toggle" onClick={() => setMounted(value => !value)}>Toggle lab</button><button data-testid="click" onClick={() => setClicks(value => value + 1)}>Respond {clicks}</button>
    <button data-testid="enabled" onClick={() => setEnabled(value => !value)}>Enable calculation</button>{mounted && <Lab input={input} enabled={enabled} />}</main>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><Harness /></StrictMode>);
