export function createBrownianBridgeWorker():Worker{return new Worker(new URL('./learning-brownian-bridge.worker.ts',import.meta.url),{type:'module',name:'brownian-bridge-calculation'});}
export function createBrownianBridgeValidationWorker():Worker{return new Worker(new URL('./learning-brownian-bridge-validation.worker.ts',import.meta.url),{type:'module',name:'brownian-bridge-validation'});}
