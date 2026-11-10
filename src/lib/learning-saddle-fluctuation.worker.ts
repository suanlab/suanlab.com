import {evaluateSaddleFluctuationRequest} from './learning-saddle-fluctuation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateSaddleFluctuationRequest(event.data,message=>scope.postMessage(message)));
