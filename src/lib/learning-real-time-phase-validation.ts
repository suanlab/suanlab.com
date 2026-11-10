/** Worker-side whole-response gate. No producer engine orits arithmetic is
 * imported. The reader-safe protocol remains a metadata/type-only boundary.
 */
import {snapshotQubitWire} from './learning-qubit-wire';
import {readRealTimePhaseRequest,realTimePhaseData,realTimePhaseInputKey,realTimePhaseUnverifiedHeader,
  type RealTimePhaseValidationReceipt,type RealTimePhaseValidatedSuccess} from './learning-real-time-phase-protocol';
import {verifyRealTimeMatrixObservations} from './learning-real-time-matrix-check';
import {verifyRealTimeUnregulatedWave} from './learning-real-time-unregulated-wave-check';
import {verifyRealTimeRegulatedWave} from './learning-real-time-regulated-wave-check';
import {verifyRealTimePacketWave} from './learning-real-time-packet-wave-check';
import {verifyRealTimeFold} from './learning-real-time-fold-check';

/** Validates a successful producer envelope against its exact owned request.
 * Failure,invalid metadata,copy-budget overflow orany mathematical mismatch
 * returns null. Actual asynchronous Worker lifecycle is implemented separately.
 */
export function verifyRealTimePhaseResponse(requestValue:unknown,candidate:unknown):RealTimePhaseValidationReceipt|null {
  try {
    const parsed=readRealTimePhaseRequest(requestValue);if('failure'in parsed)return null;
    const {input,requestId}=parsed.request,owned=snapshotQubitWire(candidate);
    if(!realTimePhaseUnverifiedHeader(owned,input,requestId))return null;
    if(input.mode==='fold') {
      if(!verifyRealTimeFold(input,owned.result))return null;
    }else {
      const result=realTimePhaseData(owned.result,input.mode==='chain'?
        ['kind','input','exact','wave','inverse','formalConnectedResponse','regulatedConnectedResponse']:['kind','input','exact','wave']);
      if(!result||result.kind!==input.mode)return null;
      if(input.mode==='packet') {
        if(!verifyRealTimePacketWave(input,result.exact,result.wave))return null;
      }else {
        const wave=realTimePhaseData(result.wave,['unregulated','regulated','workingBits','observationBits']);
        if(!wave||!Object.is(wave.workingBits,8192)||!Object.is(wave.observationBits,512))return null;
        if(!verifyRealTimeMatrixObservations(input,result.exact,{inverse:result.inverse,
          formalConnectedResponse:result.formalConnectedResponse,regulatedConnectedResponse:result.regulatedConnectedResponse}))return null;
        if(!verifyRealTimeUnregulatedWave(input,result.exact,wave.unregulated))return null;
        if(!verifyRealTimeRegulatedWave(input,result.exact,wave.regulated))return null;
      }
    }
    // This cast is deliberately after every applicable mathematical gate.
    // All levels ofowned are descriptor-only copied andrecursively frozen.
    return Object.freeze({version:1,requestId,phase:'validated',kind:'real-time-phase',inputKey:realTimePhaseInputKey(input),
      fullResponseValidated:true,response:owned as RealTimePhaseValidatedSuccess});
  }catch{return null;}
}
