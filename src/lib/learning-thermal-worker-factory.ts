export function createThermalWorker(): Worker {
  return new Worker(new URL('./learning-thermal.worker.ts', import.meta.url), { type: 'module', name: 'thermal-calculation' });
}
