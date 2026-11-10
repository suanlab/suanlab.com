/** Producer entry; Worker-side only. */
import {calculateSaddleFluctuation} from './learning-saddle-fluctuation';
import {saddleFluctuationInputKey} from './learning-saddle-fluctuation-protocol';
import {readSaddleFluctuationRequest,type SaddleFluctuationStarted,type SaddleFluctuationSuccess,type SaddleFluctuationFailure} from './learning-saddle-fluctuation-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateSaddleFluctuationRequest(value:unknown,started?:(message:SaddleFluctuationStarted)=>void):SaddleFluctuationSuccess|SaddleFluctuationFailure{
 const parsed=readSaddleFluctuationRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'saddle-fluctuation'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'saddle-fluctuation',input,inputKey:saddleFluctuationInputKey(input),result:calculateSaddleFluctuation(input)})as SaddleFluctuationSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
