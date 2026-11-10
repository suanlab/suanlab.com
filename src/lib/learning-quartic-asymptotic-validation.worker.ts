import {evaluateQuarticAsymptoticValidationRequest} from './learning-quartic-asymptotic-validation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateQuarticAsymptoticValidationRequest(event.data));
