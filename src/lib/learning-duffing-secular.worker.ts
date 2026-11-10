import {evaluateDuffingSecularRequest} from './learning-duffing-secular-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateDuffingSecularRequest(event.data,message=>scope.postMessage(message)));
