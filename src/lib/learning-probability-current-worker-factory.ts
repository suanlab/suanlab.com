export function createCurrentWorker(): Worker {
  return new Worker(new URL('./learning-probability-current.worker.ts', import.meta.url), { type: 'module', name: 'probability-current-calculation' });
}
