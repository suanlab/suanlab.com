import {evaluateGaussianFieldRequest} from './learning-gaussian-field-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateGaussianFieldRequest(event.data,message=>scope.postMessage(message)));
