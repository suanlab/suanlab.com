export function createDuffingSecularWorker():Worker{return new Worker(new URL('./learning-duffing-secular.worker.ts',import.meta.url),{type:'module',name:'duffing-secular-calculation'});}
export function createDuffingSecularValidationWorker():Worker{return new Worker(new URL('./learning-duffing-secular-validation.worker.ts',import.meta.url),{type:'module',name:'duffing-secular-validation'});}
