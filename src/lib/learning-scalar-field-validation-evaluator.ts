/** Independent validator Worker entry. Does not import the producer engine. */
import {scalarFieldData,scalarFieldSuccessHeader,type ScalarFieldReceipt} from './learning-scalar-field-protocol';
import {readScalarFieldInput,scalarFieldInputKey} from './learning-scalar-field-input';
import {validateScalarFieldResult} from './learning-scalar-field-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateScalarFieldValidationRequest(value:unknown):ScalarFieldReceipt{
 let id:number|null=null;
 try{
  const d=scalarFieldData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readScalarFieldInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!scalarFieldSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateScalarFieldResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'scalar-field',inputKey:scalarFieldInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
