/** Independent validator Worker entry. Does not import the producer engine. */
import {fieldModesData,fieldModesSuccessHeader,type FieldModesReceipt} from './learning-field-modes-protocol';
import {readFieldModesInput,fieldModesInputKey} from './learning-field-modes-input';
import {validateFieldModesResult} from './learning-field-modes-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateFieldModesValidationRequest(value:unknown):FieldModesReceipt{
 let id:number|null=null;
 try{
  const d=fieldModesData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readFieldModesInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!fieldModesSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateFieldModesResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'field-modes',inputKey:fieldModesInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
