/** Calculation Worker dispatch. No imports from this module in the reader. */
import { qubitCircuitReport } from './learning-qubit-circuit-report';
import { channelReport,environmentReport,instrumentReport,deferredMeasurementReport } from './learning-qubit-channel';
import { decodeQubitInput } from './learning-qubit-decode';
import { qubitInputKey } from './learning-qubit-input';
import { readQubitRequest,type QubitResponse,type QubitStarted,type QubitSuccess,type QubitResults } from './learning-qubit-protocol';
import { prepareQubitObservations } from './learning-qubit-observations';
import { snapshotQubitWire,QubitWireLimit } from './learning-qubit-wire';
export function evaluateQubitRequest(value:unknown,started?:(message:QubitStarted)=>void):QubitResponse {
  const parsed=readQubitRequest(value);if('failure' in parsed)return parsed.failure;
  const {requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:input.kind});
  try{
    const decoded=decodeQubitInput(input);let result:unknown;
    switch(decoded.kind){
      case 'circuit':result=qubitCircuitReport(decoded.sites,decoded.gates,decoded.input);break;
      case 'channel':result=channelReport(decoded.map,decoded.state);break;
      case 'environment':result=environmentReport(decoded.state,decoded.parameter);break;
      case 'instrument':result=instrumentReport(decoded.instrument,decoded.state,decoded.correctZ);break;
      case 'deferred':result=deferredMeasurementReport(decoded.state);break;
    }
    const observations=prepareQubitObservations(input,result as QubitResults[keyof QubitResults]);
    const response={version:1,requestId,ok:true,kind:input.kind,input,inputKey:qubitInputKey(input),result,observations};
    return snapshotQubitWire(response) as QubitSuccess;
  }catch(error){return {version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
