export function createUnitaryWorker(): Worker {
  return new Worker(new URL('./learning-unitary.worker.ts', import.meta.url), { type: 'module', name: 'unitary-calculation' });
}
