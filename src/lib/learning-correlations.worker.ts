import {evaluateCorrelationRequest} from './learning-correlations-evaluator';
import {correlationRecord,isCorrelationInput} from './learning-correlations-protocol';
const scope=self as unknown as {onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>{
 const value=event.data;
 if(correlationRecord(value)&&value.version===1&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0&&isCorrelationInput(value.input))scope.postMessage({version:1,requestId:value.requestId,phase:'started',kind:value.input.kind});
 scope.postMessage(evaluateCorrelationRequest(value));
};
