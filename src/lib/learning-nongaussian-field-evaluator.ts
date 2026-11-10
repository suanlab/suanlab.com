/** Producer entry; Worker-side only. */
import {calculateNongaussianField} from './learning-nongaussian-field';
import {nongaussianFieldInputKey} from './learning-nongaussian-field-protocol';
import {readNongaussianFieldRequest,type NongaussianFieldStarted,type NongaussianFieldSuccess,type NongaussianFieldFailure} from './learning-nongaussian-field-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateNongaussianFieldRequest(value:unknown,started?:(message:NongaussianFieldStarted)=>void):NongaussianFieldSuccess|NongaussianFieldFailure{
 const parsed=readNongaussianFieldRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'nongaussian-field'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'nongaussian-field',input,inputKey:nongaussianFieldInputKey(input),result:calculateNongaussianField(input)})as NongaussianFieldSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
