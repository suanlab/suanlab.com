import {hermitianSpectrumModel} from './learning-hermitian-spectrum';
import {sequentialObservationModel} from './learning-sequential-observation';
import {observableRecord,isObservableInput,isObservableRequestEnvelope,snapshotObservableInput,observableInputKey,observableKinds,type ObservableResponse} from './learning-observable-protocol';
export function evaluateObservableRequest(value:unknown):ObservableResponse{
 const requestId=observableRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isObservableRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isObservableInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotObservableInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:observableInputKey(input)};
  return input.kind===observableKinds[0]?{...base,kind:input.kind,input,result:hermitianSpectrumModel(input.parameters)}:{...base,kind:input.kind,input,result:sequentialObservationModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
