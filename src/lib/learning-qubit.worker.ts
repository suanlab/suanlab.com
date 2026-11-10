import { evaluateQubitRequest } from './learning-qubit-evaluator';
const scope=self as unknown as {onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateQubitRequest(event.data,message=>scope.postMessage(message)));
