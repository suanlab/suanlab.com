/** Owned result attribution without loading Rayleigh/Ritz engines on the reader. */
import{quantumVariationalKinds,quantumVariationalRecord,isQuantumVariationalInput,snapshotQuantumVariationalInput,quantumVariationalInputKey,type QuantumVariationalInput}from'./learning-quantum-variational-input';
import{shape,literal,integer}from'./learning-quantum-variational-wire-schema';
import{validateRayleighResult}from'./learning-rayleigh-validation';
import{validateRitzResult}from'./learning-ritz-validation';
import{parseNumberJsonDraft}from'./learning-json-number-input';
import type{rayleighModel}from'./learning-rayleigh';
import type{ritzModel}from'./learning-ritz';
export{quantumVariationalKinds,quantumVariationalRecord,isQuantumVariationalInput,snapshotQuantumVariationalInput,quantumVariationalInputKey,type QuantumVariationalInput}from'./learning-quantum-variational-input';
export type QuantumVariationalSuccess={version:1;requestId:number;ok:true;inputKey:string}&({kind:typeof quantumVariationalKinds[0];input:Extract<QuantumVariationalInput,{kind:typeof quantumVariationalKinds[0]}>;result:ReturnType<typeof rayleighModel>}|{kind:typeof quantumVariationalKinds[1];input:Extract<QuantumVariationalInput,{kind:typeof quantumVariationalKinds[1]}>;result:ReturnType<typeof ritzModel>});
export type QuantumVariationalResponse=QuantumVariationalSuccess|{version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'calculation-failed'};
export const isQuantumVariationalRequestEnvelope=(x:unknown):x is{version:1;requestId:number;input:unknown}=>shape(x,{version:literal(1),requestId:x=>integer(x,1),input:()=>true});
export function parseQuantumVariationalInputDraft(raw:string):QuantumVariationalInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['parameters']);return isQuantumVariationalInput(v)?snapshotQuantumVariationalInput(v):null;}catch{return null;}}
export function parseQuantumVariationalStoredInput(raw:string):QuantumVariationalInput|null{try{if(typeof raw!=='string'||raw.length>32768)return null;const v=parseNumberJsonDraft(raw,['version','parameters']);return shape(v,{version:literal(1),value:isQuantumVariationalInput})?snapshotQuantumVariationalInput(v.value as QuantumVariationalInput):null;}catch{return null;}}
export function isQuantumVariationalSuccessForInput(value:unknown,input:QuantumVariationalInput,id:number):value is QuantumVariationalSuccess{
 try{if(!integer(id,1)||!isQuantumVariationalInput(input)||!shape(value,{version:literal(1),requestId:literal(id),ok:literal(true),kind:literal(input.kind),input:isQuantumVariationalInput,inputKey:literal(quantumVariationalInputKey(input)),result:quantumVariationalRecord})||quantumVariationalInputKey(value.input as QuantumVariationalInput)!==quantumVariationalInputKey(input))return false;return input.kind===quantumVariationalKinds[0]?validateRayleighResult(value.result,input.parameters):validateRitzResult(value.result,input.parameters);}catch{return false;}
}
