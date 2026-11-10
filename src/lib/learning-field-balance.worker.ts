import {evaluateFieldBalanceRequest} from './learning-field-balance-evaluator';
const scope=self as unknown as{onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>scope.postMessage(evaluateFieldBalanceRequest(event.data,message=>scope.postMessage(message)));
