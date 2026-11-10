export function createMasterWorker(): Worker {
  return new Worker(new URL('./learning-master-equation.worker.ts', import.meta.url), { type: 'module', name: 'master-equation-calculation' });
}
