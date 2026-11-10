/** Producer entry; Worker-side only. */
import {calculateBrownianBridge} from './learning-brownian-bridge';
import {brownianBridgeInputKey} from './learning-brownian-bridge-protocol';
import {readBrownianBridgeRequest,type BrownianBridgeStarted,type BrownianBridgeSuccess,type BrownianBridgeFailure} from './learning-brownian-bridge-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateBrownianBridgeRequest(value:unknown,started?:(message:BrownianBridgeStarted)=>void):BrownianBridgeSuccess|BrownianBridgeFailure{
 const parsed=readBrownianBridgeRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'brownian-bridge'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'brownian-bridge',input,inputKey:brownianBridgeInputKey(input),result:calculateBrownianBridge(input)})as BrownianBridgeSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
