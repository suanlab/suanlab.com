/** Producer entry; Worker-side only. */
import {calculateBoundaryLayer} from './learning-boundary-layer';
import {boundaryLayerInputKey} from './learning-boundary-layer-protocol';
import {readBoundaryLayerRequest,type BoundaryLayerStarted,type BoundaryLayerSuccess,type BoundaryLayerFailure} from './learning-boundary-layer-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateBoundaryLayerRequest(value:unknown,started?:(message:BoundaryLayerStarted)=>void):BoundaryLayerSuccess|BoundaryLayerFailure{
 const parsed=readBoundaryLayerRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'boundary-layer'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'boundary-layer',input,inputKey:boundaryLayerInputKey(input),result:calculateBoundaryLayer(input)})as BoundaryLayerSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
