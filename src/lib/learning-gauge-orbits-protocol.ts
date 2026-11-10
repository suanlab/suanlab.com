/** Small reader-safe transport headers; numerical validation belongs to a Worker. */
import {readGaugeOrbitInput,gaugeOrbitInputKey,type GaugeOrbitInput} from './learning-gauge-orbits-input';
import type {GaugeOrbitResult} from './learning-gauge-orbits';
export interface GaugeOrbitSuccess {version:1;requestId:number;ok:true;kind:'gauge-orbits';input:GaugeOrbitInput;inputKey:string;result:GaugeOrbitResult;}
export interface GaugeOrbitFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export interface GaugeOrbitStarted {version:1;requestId:number;phase:'started';kind:'gauge-orbits';}
export type GaugeOrbitReceipt={version:1;requestId:number;phase:'validated';kind:'gauge-orbits';inputKey:string;response:GaugeOrbitSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function gaugeOrbitData(value:unknown,keys:readonly string[]):Record<string,unknown>|null{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
 const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;out[key]=d.value;}return out;
}
export function readGaugeOrbitRequest(value:unknown):{request:{requestId:number;input:GaugeOrbitInput}}|{failure:GaugeOrbitFailure}{
 const d=gaugeOrbitData(value,['version','requestId','input']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readGaugeOrbitInput(d.input);return input?{request:{requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function gaugeOrbitSuccessHeader(value:unknown,input:GaugeOrbitInput,id:number):value is GaugeOrbitSuccess{
 const d=gaugeOrbitData(value,['version','requestId','ok','kind','input','inputKey','result']);if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!=='gauge-orbits'||d.inputKey!==gaugeOrbitInputKey(input))return false;
 const attributed=readGaugeOrbitInput(d.input);if(!attributed||gaugeOrbitInputKey(attributed)!==gaugeOrbitInputKey(input))return false;
 const result=d.result;if(!result||typeof result!=='object')return false;const own=Object.getOwnPropertyDescriptor(result,'input'),ri=own&&Object.hasOwn(own,'value')?readGaugeOrbitInput(own.value):null;
 return !!ri&&gaugeOrbitInputKey(ri)===gaugeOrbitInputKey(input);
}
