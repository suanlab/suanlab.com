/** Literal URL keeps the calculator in a separately emitted worker asset. */
export function createEnsembleWorker(): Worker {
  return new Worker(new URL('./learning-ensemble.worker.ts', import.meta.url), { type: 'module', name: 'ensemble-calculation' });
}
