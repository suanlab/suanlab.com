/** Independent validator Worker entry. Does not import the producer engine. */
import {duffingSecularData,duffingSecularSuccessHeader,type DuffingSecularReceipt} from './learning-duffing-secular-protocol';
import {readDuffingSecularInput} from './learning-duffing-secular-input';
import {duffingSecularInputKey} from './learning-duffing-secular-protocol';
import {validateDuffingSecularResult} from './learning-duffing-secular-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateDuffingSecularValidationRequest(value:unknown):DuffingSecularReceipt{
 let id:number|null=null;
 try{
  const d=duffingSecularData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readDuffingSecularInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!duffingSecularSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateDuffingSecularResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'duffing-secular',inputKey:duffingSecularInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
