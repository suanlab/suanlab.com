export function createFieldModesWorker():Worker{return new Worker(new URL('./learning-field-modes.worker.ts',import.meta.url),{type:'module',name:'field-modes-calculation'});}
export function createFieldModesValidationWorker():Worker{return new Worker(new URL('./learning-field-modes-validation.worker.ts',import.meta.url),{type:'module',name:'field-modes-validation'});}
