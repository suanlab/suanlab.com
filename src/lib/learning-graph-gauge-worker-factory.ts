export function createGraphGaugeWorker():Worker{return new Worker(new URL('./learning-graph-gauge.worker.ts',import.meta.url),{type:'module',name:'graph-gauge-calculation'});}
export function createGraphGaugeValidationWorker():Worker{return new Worker(new URL('./learning-graph-gauge-validation.worker.ts',import.meta.url),{type:'module',name:'graph-gauge-validation'});}
