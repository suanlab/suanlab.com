/** Independent validator Worker entry. Does not import the producer engine. */
import {regularPerturbationData,regularPerturbationSuccessHeader,type RegularPerturbationReceipt} from './learning-regular-perturbation-protocol';
import {readRegularPerturbationInput} from './learning-regular-perturbation-input';
import {regularPerturbationInputKey} from './learning-regular-perturbation-protocol';
import {validateRegularPerturbationResult} from './learning-regular-perturbation-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateRegularPerturbationValidationRequest(value:unknown):RegularPerturbationReceipt{
 let id:number|null=null;
 try{
  const d=regularPerturbationData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readRegularPerturbationInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!regularPerturbationSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateRegularPerturbationResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'regular-perturbation',inputKey:regularPerturbationInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
