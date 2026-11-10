/** Worker-only dispatch: UI modules load transport checks, not these models. */
import {masterTwoStateModel} from './learning-master-two-state';
import {uniformizationModel} from './learning-master-uniformization';
import {holdingModel} from './learning-master-holding';
import {masterRecord,masterKinds,isMasterInput,isMasterRequestEnvelope,snapshotMasterInput,masterInputKey,type MasterResponse} from './learning-master-equation-protocol';
export function evaluateMasterRequest(value:unknown):MasterResponse{
 const requestId=masterRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isMasterRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isMasterInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotMasterInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:masterInputKey(input)};
  if(input.kind===masterKinds[0])return{...base,kind:input.kind,input,result:masterTwoStateModel(input.parameters)};
  if(input.kind===masterKinds[1])return{...base,kind:input.kind,input,result:uniformizationModel(input.parameters)};
  return{...base,kind:input.kind,input,result:holdingModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
