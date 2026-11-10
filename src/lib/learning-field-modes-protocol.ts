/** Small reader-safe transport headers; numerical validation belongs to a Worker. */
import {readFieldModesInput,fieldModesInputKey,type FieldModesInput} from './learning-field-modes-input';
import type {FieldModesResult} from './learning-field-modes';
export interface FieldModesSuccess {version:1;requestId:number;ok:true;kind:'field-modes';input:FieldModesInput;inputKey:string;result:FieldModesResult;}
export interface FieldModesFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export interface FieldModesStarted {version:1;requestId:number;phase:'started';kind:'field-modes';}
export type FieldModesReceipt={version:1;requestId:number;phase:'validated';kind:'field-modes';inputKey:string;response:FieldModesSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function fieldModesData(value:unknown,keys:readonly string[]):Record<string,unknown>|null{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
 const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;out[key]=d.value;}return out;
}
export function readFieldModesRequest(value:unknown):{request:{requestId:number;input:FieldModesInput}}|{failure:FieldModesFailure}{
 const d=fieldModesData(value,['version','requestId','input']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readFieldModesInput(d.input);return input?{request:{requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function fieldModesSuccessHeader(value:unknown,input:FieldModesInput,id:number):value is FieldModesSuccess{
 const d=fieldModesData(value,['version','requestId','ok','kind','input','inputKey','result']);if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!=='field-modes'||d.inputKey!==fieldModesInputKey(input))return false;
 const attributed=readFieldModesInput(d.input);if(!attributed||fieldModesInputKey(attributed)!==fieldModesInputKey(input))return false;
 const result=d.result;if(!result||typeof result!=='object')return false;const own=Object.getOwnPropertyDescriptor(result,'input'),ri=own&&Object.hasOwn(own,'value')?readFieldModesInput(own.value):null;
 return !!ri&&fieldModesInputKey(ri)===fieldModesInputKey(input);
}
