export function createSynchronizationWorker(): Worker {
  return new Worker(new URL('./learning-synchronization.worker.ts', import.meta.url), { type: 'module', name: 'synchronization-calculation' });
}
