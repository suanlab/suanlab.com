/** Producer entry; Worker-side only. */
import {calculateDuffingSecular} from './learning-duffing-secular';
import {duffingSecularInputKey} from './learning-duffing-secular-protocol';
import {readDuffingSecularRequest,type DuffingSecularStarted,type DuffingSecularSuccess,type DuffingSecularFailure} from './learning-duffing-secular-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateDuffingSecularRequest(value:unknown,started?:(message:DuffingSecularStarted)=>void):DuffingSecularSuccess|DuffingSecularFailure{
 const parsed=readDuffingSecularRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'duffing-secular'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'duffing-secular',input,inputKey:duffingSecularInputKey(input),result:calculateDuffingSecular(input)})as DuffingSecularSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
