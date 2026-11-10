export function createHilbertWorker(): Worker {
  return new Worker(new URL('./learning-hilbert.worker.ts', import.meta.url), { type: 'module', name: 'hilbert-calculation' });
}
