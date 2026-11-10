export function createScalarFieldWorker():Worker{return new Worker(new URL('./learning-scalar-field.worker.ts',import.meta.url),{type:'module',name:'scalar-field-calculation'});}
export function createScalarFieldValidationWorker():Worker{return new Worker(new URL('./learning-scalar-field-validation.worker.ts',import.meta.url),{type:'module',name:'scalar-field-validation'});}
