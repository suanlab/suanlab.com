export function createThermalFieldWorker(): Worker {
  return new Worker(new URL('./learning-thermal-field.worker.ts', import.meta.url), { type: 'module', name: 'thermal-field-calculation' });
}
export function createThermalFieldValidationWorker(): Worker {
  return new Worker(new URL('./learning-thermal-field-validation.worker.ts', import.meta.url), { type: 'module', name: 'thermal-field-validation' });
}
