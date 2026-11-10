/** Entire canonical-fold mathematical response check. Chain/packet protocol
 * composition andvalidated receipts remain separate work.
 */
import {snapshotQubitWire} from './learning-qubit-wire';
import {readRealTimePhaseInput} from './learning-real-time-phase-input';
import {realTimePhaseInputKey} from './learning-real-time-phase-protocol';
import {CheckerRational as Q,checkerCanonicalReader} from './learning-real-time-validation-rational';
import {checkerRealTimeFoldReference,checkerRealTimeAiryTerms,type CheckerAiryMethod} from './learning-real-time-checker-airy';
import {createRealTimeObservationCheck,checkerObservationRecord as record} from './learning-real-time-observation-check';
const Z=BigInt(0),O=BigInt(1);
export function verifyRealTimeFold(source:unknown,value:unknown) {
  try {
    const input=readRealTimePhaseInput(source);if(!input||input.mode!=='fold')return null;
    const owned=snapshotQubitWire(value),data=record(owned,['kind','input','method','regime','argument','phaseScale','airy','scaledAiry','integral','scaledIntegral','negativeExponential','logMagnitude','scaledSign','rawValueBelowObservationQuantum','stationaryPoints','complexStationaryPoints','stationaryApproximation','terms','workingBits','observationBits','relativeLogResolved','exactZeroClaimed']);
    const attributed=readRealTimePhaseInput(data.input);
    if(!attributed||realTimePhaseInputKey(attributed)!==realTimePhaseInputKey(input)||data.kind!=='canonical-fold'||
      !Object.is(data.workingBits,4096)||!Object.is(data.observationBits,512)||data.exactZeroClaimed!==false)throw Error('Fold metadata');
    const method=data.method as CheckerAiryMethod;
    if(!checkerRealTimeAiryTerms(input,method,data.terms as number))throw Error('Fold truncation certificate');
    const reference=checkerRealTimeFoldReference(input,method),ctx=reference.ctx,check=createRealTimeObservationCheck(ctx),read=checkerCanonicalReader();
    const u=Q.input(input.control),positiveArgument=u.n<Z;
    if(data.regime!==(u.n>Z?'two-real-stationary-points':u.n===Z?'merged-stationary-point':'no-real-stationary-points'))throw Error('Fold regime');
    for(const key of ['argument','phaseScale','airy','scaledAiry','integral','scaledIntegral','negativeExponential']as const)check.range(data[key],reference[key]);
    const scaled=record(data.scaledIntegral,['lower','upper']),lo=read(scaled.lower),hi=read(scaled.upper);
    const sign=lo.n>Z?'positive':hi.n<Z?'negative':'unresolved';
    if(data.scaledSign!==sign||data.relativeLogResolved!==(sign!=='unresolved'))throw Error('Fold displayed sign/log annotation');
    if(sign==='unresolved'){if(data.logMagnitude!==null)throw Error('Unresolved fold cannot certify log');}
    else {if(!reference.logMagnitude)throw Error('Independent fold log unresolved');check.range(data.logMagnitude,reference.logMagnitude);}
    if(typeof data.rawValueBelowObservationQuantum!=='boolean')throw Error('Fold raw annotation');
    if(!positiveArgument){if(data.rawValueBelowObservationQuantum)throw Error('Oscillatory fold cannot use decay annotation');}
    else {
      const quantum=O<<BigInt(ctx.bits-512);
      // True is a certified below-quantum claim. False is permitted in the
      // narrow threshold uncertainty band of the4096-bit production bound.
      // Both central-origin growth andasymptotic remainder give raw width
      // <2^-1024 throughout the central/overlap/real-asymptotic domains.
      if(data.rawValueBelowObservationQuantum) {if(reference.integral.upper>=quantum)throw Error('Unproven raw underflow');}
      else if(reference.integral.upper<quantum-(O<<BigInt(ctx.bits-1024)))throw Error('Missing raw underflow annotation');
      for(const key of ['airy','scaledAiry','integral','scaledIntegral']as const) {
        const r=record(data[key],['lower','upper']);if(read(r.lower).n<Z)throw Error('Positive Airy display');
      }
    }
    const array=(x:unknown,n:number):unknown[]=>{if(!Array.isArray(x)||x.length!==n)throw Error('Fold stationary shape');return x;};
    array(data.stationaryPoints,reference.stationaryPoints.length).forEach((x,i)=>{
      const point=record(x,['position','action','curvature']);
      for(const key of ['position','action','curvature']as const)check.range(point[key],reference.stationaryPoints[i][key]);
    });
    array(data.complexStationaryPoints,reference.complexStationaryPoints.length).forEach((x,i)=>{
      const point=record(x,['position','action','curvature','contributesToRealContour']),expected=reference.complexStationaryPoints[i];
      if(point.contributesToRealContour!==expected.contributesToRealContour)throw Error('Fold complex contour');
      for(const key of ['position','action','curvature']as const)check.complex(point[key],expected[key]);
    });
    if(reference.stationaryApproximation)check.range(data.stationaryApproximation,reference.stationaryApproximation);
    else if(data.stationaryApproximation!==null)throw Error('Fold stationary approximation domain');
    return Object.freeze({scope:'canonical-fold-response-only' as const,fullResponseValidated:false as const,response:owned});
  }catch{return null;}
}
