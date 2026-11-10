/** Independent validator Worker entry. Does not import the producer engine. */
import {nongaussianFieldData,nongaussianFieldSuccessHeader,type NongaussianFieldReceipt} from './learning-nongaussian-field-protocol';
import {readNongaussianFieldInput} from './learning-nongaussian-field-input';
import {nongaussianFieldInputKey} from './learning-nongaussian-field-protocol';
import {validateNongaussianFieldResult} from './learning-nongaussian-field-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateNongaussianFieldValidationRequest(value:unknown):NongaussianFieldReceipt{
 let id:number|null=null;
 try{
  const d=nongaussianFieldData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readNongaussianFieldInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!nongaussianFieldSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateNongaussianFieldResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'nongaussian-field',inputKey:nongaussianFieldInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
