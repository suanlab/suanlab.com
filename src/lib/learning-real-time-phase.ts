/** Complete finite numerical producer. Independent response validation,
 * Worker lifecycle and reader integration are separate stages.
 */
import {readRealTimePhaseInput} from './learning-real-time-phase-input';
import {buildRealTimeChainExact,buildRealTimePacketExact} from './learning-real-time-phase-exact';
import {realTimeChainCoreWire,realTimePacketCoreWire} from './learning-real-time-phase-core-wire';
import {observeRealTimeChainWave,observeRealTimePacketWave} from './learning-real-time-wave-observations';
import {packRealTimeSymmetricReal,packRealTimeSymmetricComplex} from './learning-real-time-matrix-observations';
import {calculateRealTimeFold} from './learning-real-time-airy';
export function calculateRealTimePhase(value:unknown) {
  const input=readRealTimePhaseInput(value);if(!input)throw new RangeError('Owned real-time input required');
  if(input.mode==='fold')return calculateRealTimeFold(input);
  if(input.mode==='packet') {
    const r=buildRealTimePacketExact(input);
    return{kind:'packet' as const,input,exact:realTimePacketCoreWire(r),wave:observeRealTimePacketWave(r)};
  }
  const r=buildRealTimeChainExact(input),wave=observeRealTimeChainWave(r);
  return{kind:'chain' as const,input,exact:realTimeChainCoreWire(r),wave,
    inverse:r.inertia.zero?null:packRealTimeSymmetricReal(r.d,r.inverseEntry),
    formalConnectedResponse:{interpretation:'complex-generating-response-not-probability-covariance' as const,inverseMultiplier:input.hbar,imaginary:true},
    regulatedConnectedResponse:r.regulator?packRealTimeSymmetricComplex(r.d,r.regulator.covarianceEntry):null};
}
