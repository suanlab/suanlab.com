/** Producer entry; Worker-side only. */
import {calculateRegularPerturbation} from './learning-regular-perturbation';
import {regularPerturbationInputKey} from './learning-regular-perturbation-protocol';
import {readRegularPerturbationRequest,type RegularPerturbationStarted,type RegularPerturbationSuccess,type RegularPerturbationFailure} from './learning-regular-perturbation-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateRegularPerturbationRequest(value:unknown,started?:(message:RegularPerturbationStarted)=>void):RegularPerturbationSuccess|RegularPerturbationFailure{
 const parsed=readRegularPerturbationRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'regular-perturbation'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'regular-perturbation',input,inputKey:regularPerturbationInputKey(input),result:calculateRegularPerturbation(input)})as RegularPerturbationSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
