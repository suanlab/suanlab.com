export function createUncertaintyWorker(): Worker {
  return new Worker(new URL('./learning-uncertainty.worker.ts', import.meta.url), { type: 'module', name: 'uncertainty-calculation' });
}
