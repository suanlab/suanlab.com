/** Calculation Worker adapter. A successful calculation is still unverified. */
import {calculateRealTimePhase} from './learning-real-time-phase';
import {readRealTimePhaseRequest,realTimePhaseInputKey,type RealTimePhaseStarted,type RealTimePhaseUnverifiedSuccess,type RealTimePhaseFailure} from './learning-real-time-phase-protocol';
import {snapshotQubitWire,QubitWireLimit} from './learning-qubit-wire';
export function evaluateRealTimePhaseRequest(value:unknown,started?:(message:RealTimePhaseStarted)=>void):RealTimePhaseUnverifiedSuccess|RealTimePhaseFailure {
  const parsed=readRealTimePhaseRequest(value);if('failure'in parsed)return parsed.failure;
  const{input,requestId}=parsed.request;
  try {
    started?.({version:1,requestId,phase:'started',kind:'real-time-phase'});
    return snapshotQubitWire({version:1,requestId,ok:true,kind:'real-time-phase',input,inputKey:realTimePhaseInputKey(input),result:calculateRealTimePhase(input)})as RealTimePhaseUnverifiedSuccess;
  }catch(error){return{version:1,requestId,ok:false,error:error instanceof QubitWireLimit?'resource-limit':'calculation-failed'};}
}
