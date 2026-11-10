/** Independent validator Worker entry. Does not import the producer engine. */
import {gaussianFieldData,gaussianFieldSuccessHeader,type GaussianFieldReceipt} from './learning-gaussian-field-protocol';
import {readGaussianFieldInput} from './learning-gaussian-field-input';
import {gaussianFieldInputKey} from './learning-gaussian-field-protocol';
import {validateGaussianFieldResult} from './learning-gaussian-field-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGaussianFieldValidationRequest(value:unknown):GaussianFieldReceipt{
 let id:number|null=null;
 try{
  const d=gaussianFieldData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readGaussianFieldInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!gaussianFieldSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateGaussianFieldResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'gaussian-field',inputKey:gaussianFieldInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
