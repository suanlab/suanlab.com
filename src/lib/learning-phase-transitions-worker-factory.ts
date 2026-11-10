export function createPhaseWorker(): Worker {
  return new Worker(new URL('./learning-phase-transitions.worker.ts', import.meta.url), { type: 'module', name: 'phase-transitions-calculation' });
}
