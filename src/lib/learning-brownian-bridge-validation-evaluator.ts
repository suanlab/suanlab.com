/** Independent validator Worker entry. Does not import the producer engine. */
import {brownianBridgeData,brownianBridgeSuccessHeader,type BrownianBridgeReceipt} from './learning-brownian-bridge-protocol';
import {readBrownianBridgeInput} from './learning-brownian-bridge-input';
import {brownianBridgeInputKey} from './learning-brownian-bridge-protocol';
import {validateBrownianBridgeResult} from './learning-brownian-bridge-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateBrownianBridgeValidationRequest(value:unknown):BrownianBridgeReceipt{
 let id:number|null=null;
 try{
  const d=brownianBridgeData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readBrownianBridgeInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!brownianBridgeSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateBrownianBridgeResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'brownian-bridge',inputKey:brownianBridgeInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
