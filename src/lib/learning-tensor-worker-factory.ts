export function createTensorWorker(): Worker {
  return new Worker(new URL('./learning-tensor.worker.ts', import.meta.url), { type: 'module', name: 'tensor-calculation' });
}
