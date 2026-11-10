/** Calculation Worker entry; never imported in a reader component. */
import {calculateQuantumInformation} from './learning-quantum-information-calculation';
import {quantumInformationInputKey} from './learning-quantum-information-input';
import {readInformationRequest,type InformationSuccess,type InformationFailure,type InformationStarted} from './learning-quantum-information-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateInformationRequest(value:unknown,started?:(message:InformationStarted)=>void):InformationSuccess|InformationFailure {
 const parsed=readInformationRequest(value);if('failure'in parsed)return parsed.failure;
 const {requestId,input}=parsed.request;started?.({version:1,requestId,phase:'started',kind:input.kind});
 try {
  const calculation=calculateQuantumInformation(input);
  return snapshotQubitWire({version:1,requestId,ok:true,kind:input.kind,input,inputKey:quantumInformationInputKey(input),result:calculation.result})as InformationSuccess;
 }catch(error){return {version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':error instanceof RangeError?'invalid-input':'calculation-failed'};}
}
