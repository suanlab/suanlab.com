export function createBoundaryLayerWorker():Worker{return new Worker(new URL('./learning-boundary-layer.worker.ts',import.meta.url),{type:'module',name:'boundary-layer-calculation'});}
export function createBoundaryLayerValidationWorker():Worker{return new Worker(new URL('./learning-boundary-layer-validation.worker.ts',import.meta.url),{type:'module',name:'boundary-layer-validation'});}
