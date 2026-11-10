export function createFiniteSizeWorker(): Worker {
  return new Worker(new URL('./learning-finite-size.worker.ts', import.meta.url), { type: 'module', name: 'finite-size-calculation' });
}
