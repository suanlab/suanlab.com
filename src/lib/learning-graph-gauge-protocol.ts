/** Small reader-safe transport headers; numerical validation belongs to a Worker. */
import {readGraphGaugeInput,graphGaugeInputKey,type GraphGaugeInput} from './learning-graph-gauge-input';
import type {GraphGaugeResult} from './learning-graph-gauge';
export interface GraphGaugeSuccess {version:1;requestId:number;ok:true;kind:'graph-gauge';input:GraphGaugeInput;inputKey:string;result:GraphGaugeResult;}
export interface GraphGaugeFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export interface GraphGaugeStarted {version:1;requestId:number;phase:'started';kind:'graph-gauge';}
export type GraphGaugeReceipt={version:1;requestId:number;phase:'validated';kind:'graph-gauge';inputKey:string;response:GraphGaugeSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function graphGaugeData(value:unknown,keys:readonly string[]):Record<string,unknown>|null{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
 const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;out[key]=d.value;}return out;
}
export function readGraphGaugeRequest(value:unknown):{request:{requestId:number;input:GraphGaugeInput}}|{failure:GraphGaugeFailure}{
 const d=graphGaugeData(value,['version','requestId','input']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readGraphGaugeInput(d.input);return input?{request:{requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function graphGaugeSuccessHeader(value:unknown,input:GraphGaugeInput,id:number):value is GraphGaugeSuccess{
 const d=graphGaugeData(value,['version','requestId','ok','kind','input','inputKey','result']);if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!=='graph-gauge'||d.inputKey!==graphGaugeInputKey(input))return false;
 const attributed=readGraphGaugeInput(d.input);if(!attributed||graphGaugeInputKey(attributed)!==graphGaugeInputKey(input))return false;
 const result=d.result;if(!result||typeof result!=='object')return false;const own=Object.getOwnPropertyDescriptor(result,'input'),ri=own&&Object.hasOwn(own,'value')?readGraphGaugeInput(own.value):null;
 return !!ri&&graphGaugeInputKey(ri)===graphGaugeInputKey(input);
}
