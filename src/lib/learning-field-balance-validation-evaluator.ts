/** Independent validator Worker entry. Does not import the producer engine. */
import {fieldBalanceData,fieldBalanceSuccessHeader,type FieldBalanceReceipt} from './learning-field-balance-protocol';
import {readFieldBalanceInput,fieldBalanceInputKey} from './learning-field-balance-input';
import {validateFieldBalanceResult} from './learning-field-balance-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateFieldBalanceValidationRequest(value:unknown):FieldBalanceReceipt{
 let id:number|null=null;
 try{
  const d=fieldBalanceData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readFieldBalanceInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!fieldBalanceSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateFieldBalanceResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'field-balance',inputKey:fieldBalanceInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
