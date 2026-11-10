/** Producer entry; Worker-side only. */
import {calculateGraphGauss} from './learning-graph-gauss';
import {graphGaussInputKey} from './learning-graph-gauss-input';
import {readGraphGaussRequest,type GraphGaussStarted,type GraphGaussSuccess,type GraphGaussFailure} from './learning-graph-gauss-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGraphGaussRequest(value:unknown,started?:(message:GraphGaussStarted)=>void):GraphGaussSuccess|GraphGaussFailure{
 const parsed=readGraphGaussRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'graph-gauss'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'graph-gauss',input,inputKey:graphGaussInputKey(input),result:calculateGraphGauss(input)})as GraphGaussSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
