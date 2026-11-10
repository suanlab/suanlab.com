export function createReversalWorker(): Worker {
  return new Worker(new URL('./learning-cycle-reversal.worker.ts', import.meta.url), { type: 'module', name: 'cycle-reversal-calculation' });
}
