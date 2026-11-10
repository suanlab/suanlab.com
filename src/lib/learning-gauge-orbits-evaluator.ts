/** Producer entry; Worker-side only. */
import {calculateGaugeOrbits} from './learning-gauge-orbits';
import {gaugeOrbitInputKey} from './learning-gauge-orbits-input';
import {readGaugeOrbitRequest,type GaugeOrbitStarted,type GaugeOrbitSuccess,type GaugeOrbitFailure} from './learning-gauge-orbits-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateGaugeOrbitRequest(value:unknown,started?:(message:GaugeOrbitStarted)=>void):GaugeOrbitSuccess|GaugeOrbitFailure{
 const parsed=readGaugeOrbitRequest(value);if('failure'in parsed)return parsed.failure;const{requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:'gauge-orbits'});
 try{return snapshotQubitWire({version:1,requestId,ok:true,kind:'gauge-orbits',input,inputKey:gaugeOrbitInputKey(input),result:calculateGaugeOrbits(input)})as GaugeOrbitSuccess;}
 catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
