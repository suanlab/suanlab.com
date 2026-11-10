import {densityEnsembleModel} from './learning-density-ensemble';
import {densitySpectrumModel} from './learning-density-spectrum';
import {densityRecord,isDensityInput,isDensityRequestEnvelope,snapshotDensityInput,densityInputKey,densityKinds,type DensityResponse} from './learning-density-protocol';
export function evaluateDensityRequest(value:unknown):DensityResponse{
 const requestId=densityRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isDensityRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isDensityInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotDensityInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:densityInputKey(input)};
  return input.kind===densityKinds[0]?{...base,kind:input.kind,input,result:densityEnsembleModel(input.parameters)}:{...base,kind:input.kind,input,result:densitySpectrumModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
