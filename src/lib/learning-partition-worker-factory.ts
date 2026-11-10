export function createPartitionWorker(): Worker {
  return new Worker(new URL('./learning-partition.worker.ts', import.meta.url), { type: 'module', name: 'partition-calculation' });
}
