export function createLrWorker(): Worker {
  return new Worker(new URL('./learning-linear-response.worker.ts', import.meta.url), { type: 'module', name: 'linear-response-calculation' });
}
