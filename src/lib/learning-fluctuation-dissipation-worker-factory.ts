export function createFdtWorker(): Worker {
  return new Worker(new URL('./learning-fluctuation-dissipation.worker.ts', import.meta.url), { type: 'module', name: 'fluctuation-dissipation-calculation' });
}
