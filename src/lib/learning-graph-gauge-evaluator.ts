/** Producer entry; Worker-side only. */
import {calculateGraphGauge} from './learning-graph-gauge';
import {graphGaugeInputKey} from './learning-graph-gauge-input';
import {readGraphGaugeRequest,type GraphGaugeStarted,type GraphGaugeSuccess,type GraphGaugeFailure} from './learning-graph-gauge-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGraphGaugeRequest(value:unknown,started?:(message:GraphGaugeStarted)=>void):GraphGaugeSuccess|GraphGaugeFailure{
 const parsed=readGraphGaugeRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'graph-gauge'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'graph-gauge',input,inputKey:graphGaugeInputKey(input),result:calculateGraphGauge(input)})as GraphGaugeSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
