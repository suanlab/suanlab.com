/** Independent validator Worker entry. Does not import the producer engine. */
import {boundaryLayerData,boundaryLayerSuccessHeader,type BoundaryLayerReceipt} from './learning-boundary-layer-protocol';
import {readBoundaryLayerInput} from './learning-boundary-layer-input';
import {boundaryLayerInputKey} from './learning-boundary-layer-protocol';
import {validateBoundaryLayerResult} from './learning-boundary-layer-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateBoundaryLayerValidationRequest(value:unknown):BoundaryLayerReceipt{
 let id:number|null=null;
 try{
  const d=boundaryLayerData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readBoundaryLayerInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!boundaryLayerSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateBoundaryLayerResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'boundary-layer',inputKey:boundaryLayerInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
