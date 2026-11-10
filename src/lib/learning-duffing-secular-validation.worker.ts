import {evaluateDuffingSecularValidationRequest} from './learning-duffing-secular-validation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateDuffingSecularValidationRequest(event.data));
