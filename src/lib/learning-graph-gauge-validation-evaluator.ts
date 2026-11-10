/** Independent validator Worker entry. Does not import the producer engine. */
import {graphGaugeData,graphGaugeSuccessHeader,type GraphGaugeReceipt} from './learning-graph-gauge-protocol';
import {readGraphGaugeInput,graphGaugeInputKey} from './learning-graph-gauge-input';
import {validateGraphGaugeResult} from './learning-graph-gauge-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGraphGaugeValidationRequest(value:unknown):GraphGaugeReceipt{
 let id:number|null=null;
 try{
  const d=graphGaugeData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readGraphGaugeInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!graphGaugeSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateGraphGaugeResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'graph-gauge',inputKey:graphGaugeInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
