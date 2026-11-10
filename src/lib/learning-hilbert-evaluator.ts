import {quantumStateModel} from './learning-quantum-state';
import {wavefunctionL2Model} from './learning-wavefunction-l2';
import {hilbertRecord,isHilbertInput,isHilbertRequestEnvelope,snapshotHilbertInput,hilbertInputKey,hilbertKinds,type HilbertResponse} from './learning-hilbert-protocol';
export function evaluateHilbertRequest(value:unknown):HilbertResponse{
 const requestId=hilbertRecord(value)&&Number.isSafeInteger(value.requestId)&&(value.requestId as number)>0?value.requestId as number:null;
 if(!isHilbertRequestEnvelope(value))return{version:1,requestId,ok:false,error:'invalid-request'};
 if(!isHilbertInput(value.input))return{version:1,requestId,ok:false,error:'invalid-input'};
 try{const input=snapshotHilbertInput(value.input),base={version:1 as const,requestId:value.requestId,ok:true as const,inputKey:hilbertInputKey(input)};
  return input.kind===hilbertKinds[0]?{...base,kind:input.kind,input,result:quantumStateModel(input.parameters)}:{...base,kind:input.kind,input,result:wavefunctionL2Model(input.parameters)};
 }catch{return{version:1,requestId,ok:false,error:'calculation-failed'};}
}
