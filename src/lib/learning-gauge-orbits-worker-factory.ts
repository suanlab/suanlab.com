export function createGaugeOrbitWorker():Worker{return new Worker(new URL('./learning-gauge-orbits.worker.ts',import.meta.url),{type:'module',name:'gauge-orbits-calculation'});}
export function createGaugeOrbitValidationWorker():Worker{return new Worker(new URL('./learning-gauge-orbits-validation.worker.ts',import.meta.url),{type:'module',name:'gauge-orbits-validation'});}
