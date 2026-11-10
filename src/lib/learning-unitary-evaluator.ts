import {spectralEvolutionModel} from './learning-spectral-evolution';
import {pulseEvolutionModel} from './learning-pulse-evolution';
import {unitaryRecord,isUnitaryInput,isUnitaryRequestEnvelope,snapshotUnitaryInput,unitaryInputKey,unitaryKinds,type UnitaryResponse} from './learning-unitary-protocol';
export function evaluateUnitaryRequest(value:unknown):UnitaryResponse{
 const requestId=unitaryRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isUnitaryRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isUnitaryInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotUnitaryInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:unitaryInputKey(input)};
  return input.kind===unitaryKinds[0]?{...base,kind:input.kind,input,result:spectralEvolutionModel(input.parameters)}:{...base,kind:input.kind,input,result:pulseEvolutionModel(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
