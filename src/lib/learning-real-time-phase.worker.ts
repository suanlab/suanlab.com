import {evaluateRealTimePhaseRequest} from './learning-real-time-phase-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateRealTimePhaseRequest(event.data,message=>scope.postMessage(message)));
