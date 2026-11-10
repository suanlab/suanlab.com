/** Reader-safe request/attribution metadata; no mathematical checker is loaded. */
import {readRealTimePhaseInput,type RealTimePhaseInput} from './learning-real-time-phase-input';
export type RealTimePhaseResult=ReturnType<typeof import('./learning-real-time-phase').calculateRealTimePhase>;
export interface RealTimePhaseUnverifiedSuccess {
  version:1;requestId:number;ok:true;kind:'real-time-phase';input:RealTimePhaseInput;inputKey:string;result:unknown;
}
type DeepReadonly<T>=T extends readonly(infer V)[]?readonly DeepReadonly<V>[]:T extends object?{readonly[K in keyof T]:DeepReadonly<T[K]>}:T;
/** Only the independent whole-response gate constructs these. A transport
 * claim orsuccessful attribution header alone never narrows unknown result.
 */
export type RealTimePhaseValidatedSuccess=DeepReadonly<Omit<RealTimePhaseUnverifiedSuccess,'result'>&{result:RealTimePhaseResult}>;
export interface RealTimePhaseValidationReceipt {
  readonly version:1;readonly requestId:number;readonly phase:'validated';readonly kind:'real-time-phase';
  readonly inputKey:string;readonly fullResponseValidated:true;readonly response:RealTimePhaseValidatedSuccess;
}
export interface RealTimePhaseFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export function realTimePhaseData(value:unknown,keys:readonly string[]):Record<string,unknown>|null {
  try {
    if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
    const data:Record<string,unknown>={};
    for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;Object.defineProperty(data,key,{value:d.value,enumerable:true});}
    return data;
  }catch{return null;}
}
export const realTimePhaseInputKey=(value:RealTimePhaseInput)=>JSON.stringify(readRealTimePhaseInput(value));
export function readRealTimePhaseRequest(value:unknown):{request:{requestId:number;input:RealTimePhaseInput}}|{failure:RealTimePhaseFailure} {
  const data=realTimePhaseData(value,['version','requestId','input']);
  if(!data||data.version!==1||typeof data.requestId!=='number'||!Number.isSafeInteger(data.requestId)||data.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
  const input=readRealTimePhaseInput(data.input);
  return input?{request:{requestId:data.requestId,input}}:{failure:{version:1,requestId:data.requestId,ok:false,error:'invalid-input'}};
}
/** Attribution only. This guard does not certify the result's mathematics. */
export function realTimePhaseUnverifiedHeader(value:unknown,input:RealTimePhaseInput,id:number):value is RealTimePhaseUnverifiedSuccess {
  try {
  if(!Number.isSafeInteger(id)||id<=0)return false;
  const data=realTimePhaseData(value,['version','requestId','ok','kind','input','inputKey','result']);
  if(!data||data.version!==1||data.requestId!==id||data.ok!==true||data.kind!=='real-time-phase'||data.inputKey!==realTimePhaseInputKey(input))return false;
  const attributed=readRealTimePhaseInput(data.input);
  if(!attributed||realTimePhaseInputKey(attributed)!==realTimePhaseInputKey(input)||!data.result||typeof data.result!=='object'||Array.isArray(data.result)||Object.getPrototypeOf(data.result)!==Object.prototype)return false;
  const descriptor=Object.getOwnPropertyDescriptor(data.result,'input');
  const resultInput=descriptor&&Object.hasOwn(descriptor,'value')?readRealTimePhaseInput(descriptor.value):null;
  return !!resultInput&&realTimePhaseInputKey(resultInput)===realTimePhaseInputKey(input);
  }catch{return false;}
}
