/** Producer entry; Worker-side only. */
import {calculateEuclideanKernel} from './learning-euclidean-kernel';
import {euclideanKernelInputKey} from './learning-euclidean-kernel-protocol';
import {readEuclideanKernelRequest,type EuclideanKernelStarted,type EuclideanKernelSuccess,type EuclideanKernelFailure} from './learning-euclidean-kernel-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateEuclideanKernelRequest(value:unknown,started?:(message:EuclideanKernelStarted)=>void):EuclideanKernelSuccess|EuclideanKernelFailure{
 const parsed=readEuclideanKernelRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'euclidean-kernel'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'euclidean-kernel',input,inputKey:euclideanKernelInputKey(input),result:calculateEuclideanKernel(input)})as EuclideanKernelSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
