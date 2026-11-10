/** Independent validator Worker entry. Does not import the producer engine. */
import {graphGaussData,graphGaussSuccessHeader,type GraphGaussReceipt} from './learning-graph-gauss-protocol';
import {readGraphGaussInput,graphGaussInputKey} from './learning-graph-gauss-input';
import {validateGraphGaussResult} from './learning-graph-gauss-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGraphGaussValidationRequest(value:unknown):GraphGaussReceipt{
 let id:number|null=null;
 try{
  const d=graphGaussData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readGraphGaussInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!graphGaussSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateGraphGaussResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'graph-gauss',inputKey:graphGaussInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
