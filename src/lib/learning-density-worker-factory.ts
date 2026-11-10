export function createDensityWorker(): Worker {
  return new Worker(new URL('./learning-density.worker.ts', import.meta.url), { type: 'module', name: 'density-calculation' });
}
