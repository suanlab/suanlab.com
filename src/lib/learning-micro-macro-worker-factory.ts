/** Literal URL form lets the site's webpack build emit a separate worker asset. */
export function createMicroMacroWorker(): Worker {
  return new Worker(new URL('./learning-micro-macro.worker.ts', import.meta.url), { type: 'module', name: 'micro-macro-calculation' });
}
