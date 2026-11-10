export function createManyBodyWorker(): Worker {
  return new Worker(new URL('./learning-many-body.worker.ts', import.meta.url), { type: 'module', name: 'many-body-calculation' });
}
