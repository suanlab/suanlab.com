/** Small reader-safe transport headers; numerical validation belongs to a Worker. */
import {readFieldBalanceInput,fieldBalanceInputKey,type FieldBalanceInput} from './learning-field-balance-input';
import type {FieldBalanceResult} from './learning-field-balance';
export interface FieldBalanceSuccess {version:1;requestId:number;ok:true;kind:'field-balance';input:FieldBalanceInput;inputKey:string;result:FieldBalanceResult;}
export interface FieldBalanceFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export interface FieldBalanceStarted {version:1;requestId:number;phase:'started';kind:'field-balance';}
export type FieldBalanceReceipt={version:1;requestId:number;phase:'validated';kind:'field-balance';inputKey:string;response:FieldBalanceSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function fieldBalanceData(value:unknown,keys:readonly string[]):Record<string,unknown>|null{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
 const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;out[key]=d.value;}return out;
}
export function readFieldBalanceRequest(value:unknown):{request:{requestId:number;input:FieldBalanceInput}}|{failure:FieldBalanceFailure}{
 const d=fieldBalanceData(value,['version','requestId','input']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readFieldBalanceInput(d.input);return input?{request:{requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function fieldBalanceSuccessHeader(value:unknown,input:FieldBalanceInput,id:number):value is FieldBalanceSuccess{
 const d=fieldBalanceData(value,['version','requestId','ok','kind','input','inputKey','result']);if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!=='field-balance'||d.inputKey!==fieldBalanceInputKey(input))return false;
 const attributed=readFieldBalanceInput(d.input);if(!attributed||fieldBalanceInputKey(attributed)!==fieldBalanceInputKey(input))return false;
 const result=d.result;if(!result||typeof result!=='object')return false;const own=Object.getOwnPropertyDescriptor(result,'input'),ri=own&&Object.hasOwn(own,'value')?readFieldBalanceInput(own.value):null;
 return !!ri&&fieldBalanceInputKey(ri)===fieldBalanceInputKey(input);
}
