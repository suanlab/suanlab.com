export function createBoltzmannGibbsWorker(): Worker {
  return new Worker(new URL('./learning-boltzmann-gibbs.worker.ts', import.meta.url), { type: 'module', name: 'boltzmann-gibbs-calculation' });
}
