/** Producer entry; Worker-side only. */
import {calculateQuarticAsymptotic} from './learning-quartic-asymptotic';
import {quarticAsymptoticInputKey} from './learning-quartic-asymptotic-protocol';
import {readQuarticAsymptoticRequest,type QuarticAsymptoticStarted,type QuarticAsymptoticSuccess,type QuarticAsymptoticFailure} from './learning-quartic-asymptotic-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateQuarticAsymptoticRequest(value:unknown,started?:(message:QuarticAsymptoticStarted)=>void):QuarticAsymptoticSuccess|QuarticAsymptoticFailure{
 const parsed=readQuarticAsymptoticRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'quartic-asymptotic'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'quartic-asymptotic',input,inputKey:quarticAsymptoticInputKey(input),result:calculateQuarticAsymptotic(input)})as QuarticAsymptoticSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
