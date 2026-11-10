export function createGaussianFieldWorker():Worker{return new Worker(new URL('./learning-gaussian-field.worker.ts',import.meta.url),{type:'module',name:'gaussian-field-calculation'});}
export function createGaussianFieldValidationWorker():Worker{return new Worker(new URL('./learning-gaussian-field-validation.worker.ts',import.meta.url),{type:'module',name:'gaussian-field-validation'});}
