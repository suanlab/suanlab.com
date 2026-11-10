/** Only this worker-side dispatcher imports numerical implementations. */
import {finiteCurieWeissModel} from './learning-curie-weiss';
import {curieWeissVariationalModel} from './learning-curie-weiss-variational';
import {landauSixthOrderModel} from './learning-landau-sixth-order';
import {phaseRecord,isPhaseInput,snapshotPhaseInput,phaseInputKey,phaseKinds,type PhaseResponse} from './learning-phase-transitions-protocol';
export function evaluatePhaseRequest(value:unknown):PhaseResponse {
 const requestId=phaseRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!phaseRecord(value)||value.version!==1||requestId===null)return {version:1,requestId,ok:false,error:'invalid-request'};
 if(!isPhaseInput(value.input))return {version:1,requestId,ok:false,error:'invalid-input'};
 try{
  const input=snapshotPhaseInput(value.input),base={version:1 as const,requestId,ok:true as const,inputKey:phaseInputKey(input)};
  if(input.kind===phaseKinds[0])return {...base,kind:input.kind,input,result:finiteCurieWeissModel(input.parameters,input.options)};
  if(input.kind===phaseKinds[1])return {...base,kind:input.kind,input,result:curieWeissVariationalModel(input.parameters,input.options)};
  return {...base,kind:input.kind,input,result:landauSixthOrderModel(input.parameters,input.options)};
 }catch{return {version:1,requestId,ok:false,error:'calculation-failed'};}
}
