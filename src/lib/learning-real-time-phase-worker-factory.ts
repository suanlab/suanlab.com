export function createRealTimePhaseWorker():Worker{return new Worker(new URL('./learning-real-time-phase.worker.ts',import.meta.url),{type:'module',name:'real-time-phase-calculation'});}
export function createRealTimePhaseValidationWorker():Worker{return new Worker(new URL('./learning-real-time-phase-validation.worker.ts',import.meta.url),{type:'module',name:'real-time-phase-validation'});}
