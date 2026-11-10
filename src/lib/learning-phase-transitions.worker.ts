import {evaluatePhaseRequest} from './learning-phase-transitions-evaluator';
import {phaseRecord,isPhaseInput} from './learning-phase-transitions-protocol';
const scope=self as unknown as {onmessage:((event:MessageEvent<unknown>)=>void)|null;postMessage(value:unknown):void};
scope.onmessage=event=>{
 const value=event.data;
 if(phaseRecord(value)&&value.version===1&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0&&isPhaseInput(value.input))scope.postMessage({version:1,requestId:value.requestId,phase:'started',kind:value.input.kind});
 scope.postMessage(evaluatePhaseRequest(value));
};
