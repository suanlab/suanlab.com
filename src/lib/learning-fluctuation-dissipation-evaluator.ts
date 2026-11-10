import {thermalOuResponseModel} from './learning-thermal-ou-response';
import {fdtRecord,isFdtInput,isFdtRequestEnvelope,snapshotFdtInput,fdtInputKey,type FdtResponse} from './learning-fluctuation-dissipation-protocol';
export function evaluateFdtRequest(v:unknown):FdtResponse {
  const requestId=fdtRecord(v)&&Number.isSafeInteger(v.requestId)&&(v.requestId as number)>0?v.requestId as number:null;
  if(!isFdtRequestEnvelope(v))return {version:1,requestId,ok:false,error:'invalid-request'};
  if(!isFdtInput(v.input))return {version:1,requestId,ok:false,error:'invalid-input'};
  try{const input=snapshotFdtInput(v.input);return{version:1,requestId:v.requestId,ok:true,kind:input.kind,input,inputKey:fdtInputKey(input),result:thermalOuResponseModel(input.parameters)};}catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
