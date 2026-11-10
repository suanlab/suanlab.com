import {evaluateScalarFieldRequest} from './learning-scalar-field-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateScalarFieldRequest(event.data,message=>scope.postMessage(message)));
