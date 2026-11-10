export function createCorrelationWorker(): Worker {
  return new Worker(new URL('./learning-correlations.worker.ts', import.meta.url), { type: 'module', name: 'correlations-calculation' });
}
