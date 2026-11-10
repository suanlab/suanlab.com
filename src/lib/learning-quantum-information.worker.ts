import {evaluateInformationRequest} from './learning-quantum-information-evaluator';
const scope=self as unknown as {onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateInformationRequest(event.data,message=>scope.postMessage(message)));
