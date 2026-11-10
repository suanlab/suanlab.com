import {evaluateNongaussianFieldRequest} from './learning-nongaussian-field-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateNongaussianFieldRequest(event.data,message=>scope.postMessage(message)));
