import {evaluateFieldModesRequest} from './learning-field-modes-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateFieldModesRequest(event.data,message=>scope.postMessage(message)));
