import {uncertaintyCovarianceModel} from './learning-uncertainty-covariance';
import {povmInstrumentModel} from './learning-povm-instrument';
import {uncertaintyRecord,isUncertaintyInput,isUncertaintyRequestEnvelope,snapshotUncertaintyInput,uncertaintyInputKey,uncertaintyKinds,type UncertaintyResponse} from './learning-uncertainty-protocol';
export function evaluateUncertaintyRequest(value:unknown):UncertaintyResponse{
 const requestId=uncertaintyRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isUncertaintyRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isUncertaintyInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotUncertaintyInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:uncertaintyInputKey(input)};
  return input.kind===uncertaintyKinds[0]?{...base,kind:input.kind,input,result:uncertaintyCovarianceModel(input.parameters)}:{...base,kind:input.kind,input,result:povmInstrumentModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
