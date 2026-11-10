/** Producer entry; Worker-side only. */
import {calculateSpatialSpectrum} from './learning-spatial-spectrum';
import {spatialSpectrumInputKey} from './learning-spatial-spectrum-protocol';
import {readSpatialSpectrumRequest,type SpatialSpectrumStarted,type SpatialSpectrumSuccess,type SpatialSpectrumFailure} from './learning-spatial-spectrum-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateSpatialSpectrumRequest(value:unknown,started?:(message:SpatialSpectrumStarted)=>void):SpatialSpectrumSuccess|SpatialSpectrumFailure{
 const parsed=readSpatialSpectrumRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'spatial-spectrum'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'spatial-spectrum',input,inputKey:spatialSpectrumInputKey(input),result:calculateSpatialSpectrum(input)})as SpatialSpectrumSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
