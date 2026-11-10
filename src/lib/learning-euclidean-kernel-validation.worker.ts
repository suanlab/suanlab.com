import {evaluateEuclideanKernelValidationRequest} from './learning-euclidean-kernel-validation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateEuclideanKernelValidationRequest(event.data));
