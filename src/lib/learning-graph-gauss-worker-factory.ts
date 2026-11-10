export function createGraphGaussWorker():Worker{return new Worker(new URL('./learning-graph-gauss.worker.ts',import.meta.url),{type:'module',name:'graph-gauss-calculation'});}
export function createGraphGaussValidationWorker():Worker{return new Worker(new URL('./learning-graph-gauss-validation.worker.ts',import.meta.url),{type:'module',name:'graph-gauss-validation'});}
