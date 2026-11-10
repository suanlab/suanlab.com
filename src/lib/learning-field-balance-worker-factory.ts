export function createFieldBalanceWorker():Worker{return new Worker(new URL('./learning-field-balance.worker.ts',import.meta.url),{type:'module',name:'field-balance-calculation'});}
export function createFieldBalanceValidationWorker():Worker{return new Worker(new URL('./learning-field-balance-validation.worker.ts',import.meta.url),{type:'module',name:'field-balance-validation'});}
