export function createBlockOrderWorker(): Worker {
  return new Worker(new URL('./learning-block-order.worker.ts', import.meta.url), { type: 'module', name: 'block-order-calculation' });
}
export function createBlockOrderValidationWorker(): Worker {
  return new Worker(new URL('./learning-block-order-validation.worker.ts', import.meta.url), { type: 'module', name: 'block-order-validation' });
}
