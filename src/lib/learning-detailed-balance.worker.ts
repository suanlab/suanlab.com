import {evaluateDetailedBalanceRequest} from './learning-detailed-balance-evaluator';
import {isDetailedBalanceRequestEnvelope,isDetailedBalanceInput} from './learning-detailed-balance-protocol';
const scope=self as unknown as {onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>{
 const value=event.data;
 if(isDetailedBalanceRequestEnvelope(value)&&isDetailedBalanceInput(value.input))scope.postMessage({version:1,requestId:value.requestId,phase:'started',kind:value.input.kind});
 scope.postMessage(evaluateDetailedBalanceRequest(value));
};
