/** Reader-safe ownership for local balances of the existing scalar lattice. */
import {readScalarFieldInput,scalarFieldInputKey,type ScalarFieldInput} from './learning-scalar-field-input';
export type FieldBalanceInput=Readonly<{field:ScalarFieldInput;share:string}>;
export function readFieldBalanceInput(value:unknown):FieldBalanceInput|null{
 try{
  if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype||Reflect.ownKeys(value).length!==2)return null;
  const fd=Object.getOwnPropertyDescriptor(value,'field'),sd=Object.getOwnPropertyDescriptor(value,'share');if(!fd||!sd||!Object.hasOwn(fd,'value')||!Object.hasOwn(sd,'value'))return null;
  const field=readScalarFieldInput(fd.value),share=sd.value;if(!field||typeof share!=='string'||share.length>40||!/^(?:0|[1-9]\d{0,17})(?:\/[1-9]\d{0,17})?$/.test(share))return null;
  const[n,d='1']=share.split('/');if(n.length>d.length||(n.length===d.length&&n>d))return null;
  return Object.freeze({field,share});
 }catch{return null;}
}
export const fieldBalanceInputKey=(input:FieldBalanceInput)=>JSON.stringify([scalarFieldInputKey(input.field),input.share]);
