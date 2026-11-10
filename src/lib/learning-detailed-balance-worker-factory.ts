export function createDetailedBalanceWorker(): Worker {
  return new Worker(new URL('./learning-detailed-balance.worker.ts', import.meta.url), { type: 'module', name: 'detailed-balance-calculation' });
}
