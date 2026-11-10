/** Isolated browser verification harness, never imported by application routes. */
import React, { useLayoutEffect } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { useBoltzmannGibbsCalculation } from '../../src/components/learning/use-boltzmann-gibbs-calculation';
import { useBoltzmannGibbsSeries } from '../../src/components/learning/use-boltzmann-gibbs-series';
import { boltzmannGibbsKinds, boltzmannGibbsPresets } from '../../src/lib/learning-boltzmann-gibbs';
import { isBoltzmannGibbsInput, boltzmannGibbsInputKey, type BoltzmannGibbsInput } from '../../src/lib/learning-boltzmann-gibbs-protocol';
interface Configuration { input: BoltzmannGibbsInput | null; series: BoltzmannGibbsInput[]; enabled: boolean; language: string; theme: string }
const metrics = { started: 0, acknowledged: 0, terminated: 0, live: 0, ticks: 0, blocked: false, wrongReadyInputs: 0, wrongSeriesInputs: 0, readyKinds: [] as string[], history: [] as { status: string; key: string | null; readyKey: string | null; seriesStatus: string }[] };
const NativeWorker = window.Worker;
window.Worker = class extends NativeWorker {
  private stopped = false;
  constructor(_url: string | URL, options?: WorkerOptions) {
    if (metrics.blocked) throw new Error('Explicit test-only worker construction failure');
    // esbuild does not emit Next's worker URL asset mapping. This harness maps
    // the unchanged factory to the separately compiled real worker entry.
    super(new URL('./worker.js', location.href), options); metrics.started++; metrics.live++;
    this.addEventListener('message', event => { if (event.data?.phase === 'started') metrics.acknowledged++; });
  }
  terminate() { if (!this.stopped) { this.stopped = true; metrics.terminated++; metrics.live--; } super.terminate(); }
};
setInterval(() => metrics.ticks++, 16);
let configuration: Configuration = { input: null, series: [], enabled: true, language: 'en', theme: 'light' }, root: Root | null = null;
function Probe(props: Configuration) {
  const calc = useBoltzmannGibbsCalculation(props.input, props.enabled), series = useBoltzmannGibbsSeries(props.series);
  const key = props.input === null ? null : isBoltzmannGibbsInput(props.input) ? boltzmannGibbsInputKey(props.input) : 'invalid-input';
  useLayoutEffect(() => {
    if (calc.response && calc.response.inputKey !== key) metrics.wrongReadyInputs++;
    if (calc.response) metrics.readyKinds.push(calc.response.kind);
    if (series.status === 'ready' && (series.responses.length !== props.series.length || series.responses.some((r, i) => r.inputKey !== boltzmannGibbsInputKey(props.series[i])))) metrics.wrongSeriesInputs++;
    metrics.history.push({ status: calc.status, key, readyKey: calc.response?.inputKey ?? null, seriesStatus: series.status });
  });
  return <main data-language={props.language} data-theme={props.theme}>
    <output id="state" data-status={calc.status} data-key={key ?? ''} data-ready-key={calc.response?.inputKey ?? ''} data-kind={calc.response?.kind ?? ''}>{calc.status}</output>
    <output id="series" data-status={series.status} data-count={series.responses.length}>{series.status}</output>
    <output id="error">{calc.error ?? ''}</output>
    <button id="retry" onClick={calc.retry}>Retry</button><button id="cancel" onClick={calc.cancel}>Cancel</button>
    <button id="series-retry" onClick={series.retry}>Retry series</button>
  </main>;
}
function render() {
  if (!root) root = createRoot(document.getElementById('root')!);
  root.render(<Probe {...configuration} />);
}
const api = {
  metrics,
  presets: boltzmannGibbsKinds.flatMap(kind => boltzmannGibbsPresets[kind].map(p => ({ id: p.id, input: p.input }))),
  setInput(value: unknown) { configuration = { ...configuration, input: value as BoltzmannGibbsInput | null }; render(); },
  setSeries(value: BoltzmannGibbsInput[]) { configuration = { ...configuration, series: value }; render(); },
  settings(language: string, theme: string, enabled = true) { configuration = { ...configuration, language, theme, enabled }; render(); },
  unmount() { root?.unmount(); root = null; },
  mount() { render(); },
};
Object.assign(window, { boltzmannGibbsProbe: api }); render();
