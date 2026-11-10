import {evaluateBoundaryLayerRequest} from './learning-boundary-layer-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateBoundaryLayerRequest(event.data,message=>scope.postMessage(message)));
