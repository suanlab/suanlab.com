export function createSpinWorker(): Worker {
  return new Worker(new URL('./learning-spin.worker.ts', import.meta.url), { type: 'module', name: 'spin-calculation' });
}
