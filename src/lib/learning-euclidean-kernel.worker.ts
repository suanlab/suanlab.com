import {evaluateEuclideanKernelRequest} from './learning-euclidean-kernel-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateEuclideanKernelRequest(event.data,message=>scope.postMessage(message)));
