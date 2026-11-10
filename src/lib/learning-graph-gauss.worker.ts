import {evaluateGraphGaussRequest} from './learning-graph-gauss-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateGraphGaussRequest(event.data,message=>scope.postMessage(message)));
