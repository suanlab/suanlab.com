/** Independent validator Worker entry. Does not import the producer engine. */
import {euclideanKernelData,euclideanKernelSuccessHeader,type EuclideanKernelReceipt} from './learning-euclidean-kernel-protocol';
import {readEuclideanKernelInput} from './learning-euclidean-kernel-input';
import {euclideanKernelInputKey} from './learning-euclidean-kernel-protocol';
import {validateEuclideanKernelResult} from './learning-euclidean-kernel-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateEuclideanKernelValidationRequest(value:unknown):EuclideanKernelReceipt{
 let id:number|null=null;
 try{
  const d=euclideanKernelData(value,['version','requestId','input','response']);if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
  id=d.requestId;const input=readEuclideanKernelInput(d.input);if(!input)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
  const owned=snapshotQubitWire(d.response);if(!euclideanKernelSuccessHeader(owned,input,id))return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  const checked=validateEuclideanKernelResult(input,owned.result);if(!checked)return{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  return{version:1,requestId:id,phase:'validated',kind:'euclidean-kernel',inputKey:euclideanKernelInputKey(input),response:{...owned,result:checked}};
 }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
