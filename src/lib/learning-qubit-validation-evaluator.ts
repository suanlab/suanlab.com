/** Independent validation Worker dispatch. Product models are type-only dependencies. */
import { verifyQubitCircuitReport } from './learning-qubit-circuit-validator';
import { verifyQubitChannelReport,verifyEnvironmentReport,verifyInstrumentReport,verifyDeferredMeasurementReport } from './learning-qubit-channel-validator';
import { decodeQubitInput } from './learning-qubit-decode';
import { qubitData,qubitInputKey,readQubitInput } from './learning-qubit-input';
import { qubitSuccessHeader,type QubitValidationReceipt,type QubitSuccess } from './learning-qubit-protocol';
import { snapshotQubitWire,QubitWireLimit } from './learning-qubit-wire';
import { validateQubitObservations } from './learning-qubit-observations-validator';
export function evaluateQubitValidationRequest(value:unknown):QubitValidationReceipt {
  let id:number|null=null;
  try{
    const d=qubitData(value,['version','requestId','input','response']);
    if(!d||d.version!==1||typeof d.requestId!=='number'||!Number.isSafeInteger(d.requestId)||d.requestId<=0)return {version:1,requestId:null,phase:'rejected',error:'invalid-request'};
    id=d.requestId;const input=readQubitInput(d.input);if(!input)return {version:1,requestId:id,phase:'rejected',error:'invalid-input'};
    const response=snapshotQubitWire(d.response);
    if(!qubitSuccessHeader(response,input,id))return {version:1,requestId:id,phase:'rejected',error:'invalid-response'};
    const decoded=decodeQubitInput(input);let valid=false;
    switch(decoded.kind){
      case 'circuit':valid=verifyQubitCircuitReport(decoded.sites,decoded.gates,decoded.input,response.result).valid;break;
      case 'channel':valid=verifyQubitChannelReport(decoded.map,decoded.state,response.result).valid;break;
      case 'environment':valid=verifyEnvironmentReport(decoded.state,decoded.parameter,response.result).valid;break;
      case 'instrument':valid=verifyInstrumentReport(decoded.instrument,decoded.state,decoded.correctZ,response.result).valid;break;
      case 'deferred':valid=verifyDeferredMeasurementReport(decoded.state,response.result).valid;break;
    }
    valid=valid&&validateQubitObservations(input,response.result,response.observations);
    return valid?{version:1,requestId:id,phase:'validated',kind:input.kind,inputKey:qubitInputKey(input),response:response as QubitSuccess}:{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  }catch(error){return {version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
