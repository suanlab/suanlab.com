export function createQuantumVariationalWorker(): Worker {
  return new Worker(new URL('./learning-quantum-variational.worker.ts', import.meta.url), { type: 'module', name: 'quantum-variational-calculation' });
}
