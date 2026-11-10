import {evaluateRealTimePhaseValidationRequest} from './learning-real-time-phase-validation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateRealTimePhaseValidationRequest(event.data,message=>scope.postMessage(message)));
