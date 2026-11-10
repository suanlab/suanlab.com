/** Small reader-safe transport headers; numerical validation belongs to a Worker. */
import {readGraphGaussInput,graphGaussInputKey,type GraphGaussInput} from './learning-graph-gauss-input';
import type {GraphGaussResult} from './learning-graph-gauss';
export interface GraphGaussSuccess {version:1;requestId:number;ok:true;kind:'graph-gauss';input:GraphGaussInput;inputKey:string;result:GraphGaussResult;}
export interface GraphGaussFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export interface GraphGaussStarted {version:1;requestId:number;phase:'started';kind:'graph-gauss';}
export type GraphGaussReceipt={version:1;requestId:number;phase:'validated';kind:'graph-gauss';inputKey:string;response:GraphGaussSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function graphGaussData(value:unknown,keys:readonly string[]):Record<string,unknown>|null{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
 const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;out[key]=d.value;}return out;
}
export function readGraphGaussRequest(value:unknown):{request:{requestId:number;input:GraphGaussInput}}|{failure:GraphGaussFailure}{
 const d=graphGaussData(value,['version','requestId','input']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readGraphGaussInput(d.input);return input?{request:{requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function graphGaussSuccessHeader(value:unknown,input:GraphGaussInput,id:number):value is GraphGaussSuccess{
 const d=graphGaussData(value,['version','requestId','ok','kind','input','inputKey','result']);if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!=='graph-gauss'||d.inputKey!==graphGaussInputKey(input))return false;
 const attributed=readGraphGaussInput(d.input);if(!attributed||graphGaussInputKey(attributed)!==graphGaussInputKey(input))return false;
 const result=d.result;if(!result||typeof result!=='object')return false;const own=Object.getOwnPropertyDescriptor(result,'input'),ri=own&&Object.hasOwn(own,'value')?readGraphGaussInput(own.value):null;
 return !!ri&&graphGaussInputKey(ri)===graphGaussInputKey(input);
}
