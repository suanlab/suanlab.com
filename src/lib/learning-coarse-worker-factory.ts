export function createCoarseWorker(): Worker {
  return new Worker(new URL('./learning-coarse.worker.ts', import.meta.url), { type: 'module', name: 'coarse-calculation' });
}
