import {markovCoarseModel} from './learning-markov-coarse';
import {linearMemoryModel} from './learning-linear-memory';
import {coarseRecord,isCoarseInput,isCoarseRequestEnvelope,snapshotCoarseInput,coarseInputKey,coarseKinds,type CoarseResponse} from './learning-coarse-protocol';
export function evaluateCoarseRequest(value:unknown):CoarseResponse{
 const requestId=coarseRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isCoarseRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isCoarseInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotCoarseInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:coarseInputKey(input)};
  return input.kind===coarseKinds[0]?{...base,kind:input.kind,input,result:markovCoarseModel(input.parameters)}:{...base,kind:input.kind,input,result:linearMemoryModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
