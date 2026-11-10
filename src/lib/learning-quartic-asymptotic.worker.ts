import {evaluateQuarticAsymptoticRequest} from './learning-quartic-asymptotic-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateQuarticAsymptoticRequest(event.data,message=>scope.postMessage(message)));
