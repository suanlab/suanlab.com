/** Worker-only numerical dispatch. */
import {ouMomentModel}from'./learning-ou-moments';
import {fokkerVolumeModel}from'./learning-fokker-volume';
import {fokkerRecord,isFokkerInput,isFokkerRequestEnvelope,snapshotFokkerInput,fokkerInputKey,type FokkerResponse}from'./learning-fokker-planck-protocol';
export function evaluateFokkerRequest(v:unknown):FokkerResponse{
 const requestId=fokkerRecord(v)&&Number.isSafeInteger(v.requestId)&&(v.requestId as number)>0?v.requestId as number:null;
 if(!isFokkerRequestEnvelope(v))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isFokkerInput(v.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotFokkerInput(v.input),base={version:1 as const,requestId:v.requestId,ok:true as const,inputKey:fokkerInputKey(input)};
  return input.kind==='ornstein-uhlenbeck-density-euler-moments'?{...base,kind:input.kind,input,result:ouMomentModel(input.parameters)}:{...base,kind:input.kind,input,result:fokkerVolumeModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
