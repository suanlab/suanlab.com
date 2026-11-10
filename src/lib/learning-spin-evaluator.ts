import {spinHalfRotationModel} from './learning-spin-half';
import {spinLadderModel} from './learning-spin-ladder';
import {spinRecord,isSpinInput,isSpinRequestEnvelope,snapshotSpinInput,spinInputKey,spinKinds,type SpinResponse} from './learning-spin-protocol';
export function evaluateSpinRequest(value:unknown):SpinResponse{
 const requestId=spinRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isSpinRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isSpinInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotSpinInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:spinInputKey(input)};
  return input.kind===spinKinds[0]?{...base,kind:input.kind,input,result:spinHalfRotationModel(input.parameters)}:{...base,kind:input.kind,input,result:spinLadderModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
