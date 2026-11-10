/** Independent validation Worker adapter. No producer engine is imported. */
import {realTimePhaseData,readRealTimePhaseRequest,type RealTimePhaseValidationOutcome,type RealTimePhaseVerificationStarted} from './learning-real-time-phase-protocol';
import {verifyRealTimePhaseResponse} from './learning-real-time-phase-validation';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateRealTimePhaseValidationRequest(value:unknown,started?:(message:RealTimePhaseVerificationStarted)=>void):RealTimePhaseValidationOutcome {
  let id:number|null=null;
  try {
    const data=realTimePhaseData(value,['version','requestId','input','response']);
    if(!data||data.version!==1||typeof data.requestId!=='number'||!Number.isSafeInteger(data.requestId)||data.requestId<=0)return{version:1,requestId:null,phase:'rejected',error:'invalid-request'};
    id=data.requestId;const request={version:1,requestId:id,input:data.input},parsed=readRealTimePhaseRequest(request);
    if('failure'in parsed)return{version:1,requestId:id,phase:'rejected',error:'invalid-input'};
    started?.({version:1,requestId:id,phase:'verification-started',kind:'real-time-phase'});
    // Distinguish copy-resource failures before the whole mathematical gate.
    const owned=snapshotQubitWire(data.response),receipt=verifyRealTimePhaseResponse(request,owned);
    return receipt??{version:1,requestId:id,phase:'rejected',error:'invalid-response'};
  }catch(error){return{version:1,requestId:id,phase:'rejected',error:error instanceof QubitWireLimit?'resource-limit':'invalid-response'};}
}
