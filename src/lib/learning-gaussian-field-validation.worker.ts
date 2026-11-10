import {evaluateGaussianFieldValidationRequest} from './learning-gaussian-field-validation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateGaussianFieldValidationRequest(event.data));
