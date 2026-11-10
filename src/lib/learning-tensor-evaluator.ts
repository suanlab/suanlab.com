import {tensorPureModel} from './learning-tensor-pure';
import {mixedEntanglementModel} from './learning-mixed-entanglement';
import {tensorRecord,isTensorInput,isTensorRequestEnvelope,snapshotTensorInput,tensorInputKey,tensorKinds,type TensorResponse} from './learning-tensor-protocol';
export function evaluateTensorRequest(value:unknown):TensorResponse{
 const requestId=tensorRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isTensorRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isTensorInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotTensorInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:tensorInputKey(input)};
  return input.kind===tensorKinds[0]?{...base,kind:input.kind,input,result:tensorPureModel(input.parameters)}:{...base,kind:input.kind,input,result:mixedEntanglementModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
