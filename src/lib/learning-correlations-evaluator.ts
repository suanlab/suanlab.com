/** Only the worker-side dispatcher imports the numerical models. */
import {binaryConnectedCorrelationsModel} from './learning-binary-correlations';
import {periodicIsingRingModel} from './learning-periodic-ising-ring';
import {ar1FiniteAverageModel} from './learning-ar1-average';
import {correlationRecord,isCorrelationInput,snapshotCorrelationInput,correlationInputKey,correlationKinds,type CorrelationResponse} from './learning-correlations-protocol';
export function evaluateCorrelationRequest(value:unknown):CorrelationResponse {
 const requestId=correlationRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!correlationRecord(value)||value.version!==1||requestId===null)return {version:1,requestId,ok:false,error:'invalid-request'};
 if(!isCorrelationInput(value.input))return {version:1,requestId,ok:false,error:'invalid-input'};
 try{
  const input=snapshotCorrelationInput(value.input),base={version:1 as const,requestId,ok:true as const,inputKey:correlationInputKey(input)};
  if(input.kind===correlationKinds[0])return {...base,kind:input.kind,input,result:binaryConnectedCorrelationsModel(input.parameters,input.options)};
  if(input.kind===correlationKinds[1])return {...base,kind:input.kind,input,result:periodicIsingRingModel(input.parameters,input.options)};
  return {...base,kind:input.kind,input,result:ar1FiniteAverageModel(input.parameters,input.options)};
 }catch{return {version:1,requestId,ok:false,error:'calculation-failed'};}
}
