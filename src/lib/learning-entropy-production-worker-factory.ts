export function createEntropyWorker(): Worker {
  return new Worker(new URL('./learning-entropy-production.worker.ts', import.meta.url), { type: 'module', name: 'entropy-production-calculation' });
}
