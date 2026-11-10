import React, { StrictMode, useEffect, useLayoutEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { useMicroMacroCalculation } from '../../src/components/learning/use-micro-macro-calculation';
import type { MicroMacroInput } from '../../src/lib/learning-micro-macro-protocol';
const binary: MicroMacroInput = { kind: 'binary-shell-multiplicity-block-counts', parameters: { sites: 4, excitations: 2, blockASites: 2, energyGap: 1 } };
const weighted: MicroMacroInput = { kind: 'finite-macro-partition-conditional-entropy', parameters: { trialWeights: [6, 2, 1, 1], targetWeights: [4, 4, 1, 1], mapping: [0, 0, 1, 1], groupCount: 2 } };
const noisy: MicroMacroInput = { kind: 'finite-noisy-macro-observation-channel', parameters: { preparationWeights: [1, 1], channelWeights: [[9, 1], [1, 9]] } };
const heavy: MicroMacroInput = { kind: noisy.kind, parameters: { preparationWeights: Array.from({ length: 16 }, (_, i) => i % 2 ? Number.MAX_VALUE / (i + 1) : Number.MIN_VALUE * (i + 1)),
  channelWeights: Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => j === i ? Number.MAX_VALUE / (i + 1) : Number.MIN_VALUE * (j + 1))) } };
const probe = { started: 0, terminated: 0, live: 0, heartbeats: 0, maxGap: 0, clicks: 0, staleResults: 0, renders: 0,
  latest: { status: '', phase: null as string | null, kind: '', inputs: 0, requestId: null as number | null, value: null as number | null } };
Object.assign(window, { microMacroProbe: probe });
const NativeWorker = window.Worker;
class TrackedWorker extends NativeWorker {
  private finished = false;
  constructor(url: string | URL, options?: WorkerOptions) { super(url, options); probe.started++; probe.live++; }
  terminate() { if (!this.finished) { this.finished = true; probe.terminated++; probe.live--; } super.terminate(); }
}
window.Worker = TrackedWorker;
let last = performance.now();
setInterval(() => { const now = performance.now(); probe.heartbeats++; probe.maxGap = Math.max(probe.maxGap, now - last); last = now; }, 16);
function Lab({ input, enabled }: { input: MicroMacroInput; enabled: boolean }) {
  // Recreate each input object every render to exercise key-based scheduling.
  const result = useMicroMacroCalculation(structuredClone(input), enabled);
  useLayoutEffect(() => {
    probe.renders++;
    let stale = false;
    const response = result.response;
    if (response) {
      stale = response.kind !== input.kind;
      if (response.kind === 'finite-noisy-macro-observation-channel' && input.kind === 'finite-noisy-macro-observation-channel')
        stale ||= JSON.stringify(response.result.parameters) !== JSON.stringify(input.parameters);
      if (response.kind === 'binary-shell-multiplicity-block-counts' && input.kind === 'binary-shell-multiplicity-block-counts') stale ||= response.result.parameters.sites !== input.parameters.sites;
    }
    if (stale) probe.staleResults++;
    probe.latest = { status: result.status, phase: result.phase, kind: input.kind, inputs: input.kind === 'finite-noisy-macro-observation-channel' ? input.parameters.preparationWeights.length : 0,
      requestId: result.requestId, value: response?.kind === 'finite-noisy-macro-observation-channel' ? response.result.mutualInformation.value : null };
  });
  return <section><p data-testid="status">{result.status}</p><button data-testid="cancel" onClick={result.cancel}>Cancel</button><button data-testid="retry" onClick={result.retry}>Retry</button></section>;
}
function Harness() {
  const [input, setInput] = useState<MicroMacroInput>(binary), [mounted, setMounted] = useState(true), [clicks, setClicks] = useState(0), [enabled, setEnabled] = useState(true);
  useEffect(() => { probe.clicks = clicks; }, [clicks]);
  return <main><button data-testid="binary" onClick={() => setInput(binary)}>Binary</button><button data-testid="weighted" onClick={() => setInput(weighted)}>Weighted</button>
    <button data-testid="noisy" onClick={() => setInput(noisy)}>Noisy</button><button data-testid="heavy" onClick={() => setInput(heavy)}>Heavy</button>
    <button data-testid="toggle" onClick={() => setMounted(value => !value)}>Toggle lab</button><button data-testid="click" onClick={() => setClicks(value => value + 1)}>Respond {clicks}</button>
    <button data-testid="enabled" onClick={() => setEnabled(value => !value)}>Enable calculation</button>{mounted && <Lab input={input} enabled={enabled} />}</main>;
}
createRoot(document.getElementById('root')!).render(<StrictMode><Harness /></StrictMode>);
