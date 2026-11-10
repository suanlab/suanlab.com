/** Small reader-safe transport headers; numerical validation belongs to a Worker. */
import {readScalarFieldInput,scalarFieldInputKey,type ScalarFieldInput} from './learning-scalar-field-input';
import type {ScalarFieldResult} from './learning-scalar-field';
export interface ScalarFieldSuccess {version:1;requestId:number;ok:true;kind:'scalar-field';input:ScalarFieldInput;inputKey:string;result:ScalarFieldResult;}
export interface ScalarFieldFailure {version:1;requestId:number|null;ok:false;error:'invalid-request'|'invalid-input'|'resource-limit'|'calculation-failed';}
export interface ScalarFieldStarted {version:1;requestId:number;phase:'started';kind:'scalar-field';}
export type ScalarFieldReceipt={version:1;requestId:number;phase:'validated';kind:'scalar-field';inputKey:string;response:ScalarFieldSuccess}|{version:1;requestId:number|null;phase:'rejected';error:'invalid-request'|'invalid-input'|'invalid-response'|'resource-limit'};
export function scalarFieldData(value:unknown,keys:readonly string[]):Record<string,unknown>|null{
 if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==keys.length)return null;
 const out:Record<string,unknown>={};for(const key of keys){const d=Object.getOwnPropertyDescriptor(value,key);if(!d||!Object.hasOwn(d,'value'))return null;out[key]=d.value;}return out;
}
export function readScalarFieldRequest(value:unknown):{request:{requestId:number;input:ScalarFieldInput}}|{failure:ScalarFieldFailure}{
 const d=scalarFieldData(value,['version','requestId','input']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{failure:{version:1,requestId:null,ok:false,error:'invalid-request'}};
 const input=readScalarFieldInput(d.input);return input?{request:{requestId:d.requestId,input}}:{failure:{version:1,requestId:d.requestId,ok:false,error:'invalid-input'}};
}
export function scalarFieldSuccessHeader(value:unknown,input:ScalarFieldInput,id:number):value is ScalarFieldSuccess{
 const d=scalarFieldData(value,['version','requestId','ok','kind','input','inputKey','result']);if(!d||d.version!==1||d.requestId!==id||d.ok!==true||d.kind!=='scalar-field'||d.inputKey!==scalarFieldInputKey(input))return false;
 const attributed=readScalarFieldInput(d.input);if(!attributed||scalarFieldInputKey(attributed)!==scalarFieldInputKey(input))return false;
 const result=d.result;if(!result||typeof result!=='object')return false;const own=Object.getOwnPropertyDescriptor(result,'input'),ri=own&&Object.hasOwn(own,'value')?readScalarFieldInput(own.value):null;
 return !!ri&&scalarFieldInputKey(ri)===scalarFieldInputKey(input);
}
