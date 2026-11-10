export function createFokkerWorker(): Worker {
  return new Worker(new URL('./learning-fokker-planck.worker.ts', import.meta.url), { type: 'module', name: 'fokker-planck-calculation' });
}
