/** Owned result transport. Independent references establish spectral/logarithmic math;
 * bounded schema and accounts attribute results,without rerunning either engine. */
import {tensorKinds,tensorRecord,isTensorInput,snapshotTensorInput,tensorInputKey,type TensorInput} from './learning-tensor-input';
import {shape,literal,integer} from './learning-tensor-wire-schema';
import {validatePureTensorResult} from './learning-tensor-pure-validation';
import {validateMixedTensorResult} from './learning-tensor-mixed-validation';
import {parseNumberJsonDraft} from './learning-json-number-input';
import type {tensorPureModel} from './learning-tensor-pure';
import type {mixedEntanglementModel} from './learning-mixed-entanglement';
export {tensorKinds,tensorRecord,isTensorInput,snapshotTensorInput,tensorInputKey,type TensorInput} from './learning-tensor-input';
export type TensorSuccess={version:1;requestId:number;ok:true;inputKey:string}&({kind:typeof tensorKinds[0];input:Extract<TensorInput,{kind:typeof tensorKinds[0]}>;result:ReturnType<typeof tensorPureModel>}|{kind:typeof tensorKinds[1];input:Extract<TensorInput,{kind:typeof tensorKinds[1]}>;result:ReturnType<typeof mixedEntanglementModel>});
export type TensorResponse=TensorSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const isTensorRequestEnvelope=(x:unknown):x is{version:1;requestId:number;input:unknown}=>shape(x,{version:literal(1),requestId:v=>integer(v,1),input:()=>true});
export function parseTensorInputDraft(raw:string):TensorInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['parameters']);return isTensorInput(v)?snapshotTensorInput(v):null;}catch{return null;}}
export function parseTensorStoredInput(raw:string):TensorInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['version','parameters']);return shape(v,{version:literal(1),value:isTensorInput})?snapshotTensorInput(v.value as TensorInput):null;}catch{return null;}}
export function isTensorSuccessForInput(value:unknown,input:TensorInput,id:number):value is TensorSuccess{
 try{
  if(!integer(id,1)||!isTensorInput(input)||!shape(value,{version:literal(1),requestId:literal(id),ok:literal(true),kind:literal(input.kind),input:isTensorInput,inputKey:literal(tensorInputKey(input)),result:tensorRecord})||tensorInputKey(value.input as TensorInput)!==tensorInputKey(input))return false;
  return input.kind===tensorKinds[0]?validatePureTensorResult(value.result,input.parameters):validateMixedTensorResult(value.result,input.parameters);
 }catch{return false;}
}
