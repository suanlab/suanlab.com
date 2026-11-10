import {evaluateInformationValidationRequest} from './learning-quantum-information-validation-evaluator';
const scope=self as unknown as {onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateInformationValidationRequest(event.data));
