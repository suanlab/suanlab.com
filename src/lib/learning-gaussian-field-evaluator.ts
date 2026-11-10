/** Producer entry; Worker-side only. */
import {calculateGaussianField} from './learning-gaussian-field';
import {gaussianFieldInputKey} from './learning-gaussian-field-protocol';
import {readGaussianFieldRequest,type GaussianFieldStarted,type GaussianFieldSuccess,type GaussianFieldFailure} from './learning-gaussian-field-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGaussianFieldRequest(value:unknown,started?:(message:GaussianFieldStarted)=>void):GaussianFieldSuccess|GaussianFieldFailure{
 const parsed=readGaussianFieldRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'gaussian-field'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'gaussian-field',input,inputKey:gaussianFieldInputKey(input),result:calculateGaussianField(input)})as GaussianFieldSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
