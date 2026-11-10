/** Independent validator Worker entry. Does not import the producer engine. */
import {saddleFluctuationData,saddleFluctuationSuccessHeader,type SaddleFluctuationReceipt} from './learning-saddle-fluctuation-protocol';
import {readSaddleFluctuationInput} from './learning-saddle-fluctuation-input';
import {saddleFluctuationInputKey} from './learning-saddle-fluctuation-protocol';
import {validateSaddleFluctuationResult} from './learning-saddle-fluctuation-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateSaddleFluctuationValidationRequest(value:unknown):SaddleFluctuationReceipt{
 let id:number|null=null;
 try{
  const d=saddleFluctuationData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readSaddleFluctuationInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!saddleFluctuationSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateSaddleFluctuationResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'saddle-fluctuation',inputKey:saddleFluctuationInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
