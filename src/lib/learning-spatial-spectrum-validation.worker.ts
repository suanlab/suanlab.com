import {evaluateSpatialSpectrumValidationRequest} from './learning-spatial-spectrum-validation-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateSpatialSpectrumValidationRequest(event.data));
