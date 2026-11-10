export function createObservableWorker(): Worker {
  return new Worker(new URL('./learning-observable.worker.ts', import.meta.url), { type: 'module', name: 'observable-calculation' });
}
