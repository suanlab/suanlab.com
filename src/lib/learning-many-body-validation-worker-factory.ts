export function createManyBodyValidationWorker(): Worker {
  return new Worker(new URL('./learning-many-body-validation.worker.ts', import.meta.url), { type: 'module', name: 'many-body-validation' });
}
