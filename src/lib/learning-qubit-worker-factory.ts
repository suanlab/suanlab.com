export function createQubitWorker():Worker {return new Worker(new URL('./learning-qubit.worker.ts',import.meta.url),{type:'module',name:'qubit-calculation'});}
export function createQubitValidationWorker():Worker {return new Worker(new URL('./learning-qubit-validation.worker.ts',import.meta.url),{type:'module',name:'qubit-validation'});}
