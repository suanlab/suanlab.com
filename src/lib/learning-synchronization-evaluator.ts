import {hopfAmplitudeModel} from './learning-hopf-amplitude';
import {pairPhaseLockingModel} from './learning-pair-phase-locking';
import {finiteKuramotoModel} from './learning-finite-kuramoto';
import {synchronizationRecord,isSynchronizationInput,isSynchronizationRequestEnvelope,snapshotSynchronizationInput,synchronizationInputKey,synchronizationKinds,type SynchronizationResponse} from './learning-synchronization-protocol';
export function evaluateSynchronizationRequest(value:unknown):SynchronizationResponse{
 const requestId=synchronizationRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isSynchronizationRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isSynchronizationInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{
  const input=snapshotSynchronizationInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:synchronizationInputKey(input)};
  if(input.kind===synchronizationKinds[0])return{...base,kind:input.kind,input,result:hopfAmplitudeModel(input.parameters)};
  if(input.kind===synchronizationKinds[1])return{...base,kind:input.kind,input,result:pairPhaseLockingModel(input.parameters)};
  return{...base,kind:input.kind,input,result:finiteKuramotoModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
