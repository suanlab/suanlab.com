/** Finite continuous-time generator: exact rational Euler values and
 * directed exponential reference. Worker-only numerical evaluation. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F} from '@/lib/learning-finite-markov-exact';
import {createTranscendentalContext} from '@/lib/learning-interval-transcendentals';
import {partitionObservationFromRange,type PartitionObservation as Observation} from '@/lib/learning-partition-exact-core';
export interface MasterTwoStateParameters {
 forwardNumerator:number; backwardNumerator:number; rateDenominator:number;
 timeNumerator:number; timeDenominator:number;
 initialOneNumerator:number; initialDenominator:number; eulerSteps:number;
}
interface Interval {lower:F;upper:F}
const Z=BigInt(0),O=BigInt(1);
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d);
const neg=(a:F)=>f(-a.n,a.d);
const sub=(a:F,b:F)=>add(a,neg(b));
const mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d);
const div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n);
const abs=(a:F)=>a.n<Z?neg(a):a;
const le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
const point=(a:F):Interval=>({lower:a,upper:a});
const shift=(a:Interval,b:F):Interval=>({lower:add(a.lower,b),upper:add(a.upper,b)});
const scale=(a:Interval,b:F):Interval=>b.n<Z?{lower:mul(a.upper,b),upper:mul(a.lower,b)}:{lower:mul(a.lower,b),upper:mul(a.upper,b)};
const subtract=(a:Interval,b:Interval):Interval=>({lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)});
const output=(q:F)=>exact(q.n,q.d);
const binary=(q:bigint)=>({coefficient:q,exponent:0});
function observation(q:Interval):Observation {
 if(q.lower.n*q.upper.d===q.upper.n*q.lower.d)return output(q.lower);
 return partitionObservationFromRange({lower:{n:binary(q.lower.n),d:binary(q.lower.d)},upper:{n:binary(q.upper.n),d:binary(q.upper.d)}});
}
const integer=(x:unknown,min:number,max=1000000):x is number=>typeof x==='number'&&Number.isInteger(x)&&x>=min&&x<=max;
export function isMasterTwoStateParameters(value:unknown):value is MasterTwoStateParameters {
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const p=value as Record<string,unknown>,keys=['forwardNumerator','backwardNumerator','rateDenominator','timeNumerator','timeDenominator','initialOneNumerator','initialDenominator','eulerSteps'];
 return Object.keys(p).length===keys.length&&keys.every(k=>Object.prototype.hasOwnProperty.call(p,k))
 &&['forwardNumerator','backwardNumerator','timeNumerator','initialOneNumerator'].every(k=>integer(p[k],0))
 &&['rateDenominator','timeDenominator','initialDenominator'].every(k=>integer(p[k],1))&&integer(p.eulerSteps,1,128)
 &&(p.forwardNumerator as number)<=5*(p.rateDenominator as number)&&(p.backwardNumerator as number)<=5*(p.rateDenominator as number)
 &&(p.timeNumerator as number)<=10*(p.timeDenominator as number)&&(p.initialOneNumerator as number)<=(p.initialDenominator as number);
}
export function masterTwoStateModel(input:MasterTwoStateParameters){
 if(!isMasterTwoStateParameters(input))throw new RangeError('Eight owned integer fields; rates0..5/s, time0..10s, initial probability0..1, denominators1..1000000 and Euler steps1..128');
 const p=Object.freeze({...input}),a=f(BigInt(p.forwardNumerator),BigInt(p.rateDenominator)),b=f(BigInt(p.backwardNumerator),BigInt(p.rateDenominator));
 const time=f(BigInt(p.timeNumerator),BigInt(p.timeDenominator)),u=f(BigInt(p.initialOneNumerator),BigInt(p.initialDenominator)),lambda=add(a,b),dt=div(time,f(BigInt(p.eulerSteps)));
 const one=f(O),zero=f(Z),ctx=createTranscendentalContext(256);
 const exponential=(x:F):Interval=>{const q=ctx.expNegativeRatio(binary(x.n),binary(x.d));return{lower:f(q.lower,ctx.scale),upper:f(q.upper,ctx.scale)}};
 const pi=lambda.n===Z?null:div(a,lambda),d=pi===null?zero:sub(u,pi),factor=sub(one,mul(lambda,dt));
 const generator=[[neg(a),a],[b,neg(b)]],eulerKernel=[[sub(one,mul(a,dt)),mul(a,dt)],[mul(b,dt),sub(one,mul(b,dt))]];
 const distribution=(x:Interval)=>[observation(shift(scale(x,f(-O)),one)),observation(x)];
 let power=one;const history=[];
 for(let step=0;step<=p.eulerSteps;step++){
  const t=mul(dt,f(BigInt(step))),q=exponential(mul(lambda,t)),deviation=pi===null?point(zero):scale(q,d),exactOne=pi===null?point(u):shift(deviation,pi),eulerOne=pi===null?u:add(pi,mul(d,power));
  const error=pi===null?point(zero):scale(subtract(point(power),q),d);
  history.push({step,time:output(t),relaxationFactor:observation(q),exactProbabilities:distribution(exactOne),eulerValues:[output(sub(one,eulerOne)),output(eulerOne)],
   signedDeviationFromUniqueStationary:pi===null?null:observation(deviation),signedEulerError:observation(error),
   forwardFlux:observation(scale(shift(scale(exactOne,f(-O)),one),a)),backwardFlux:observation(scale(exactOne,b)),netFlux:pi===null?output(zero):observation(scale(deviation,neg(lambda))),
   exactNormalizationResidual:output(zero),eulerNormalizationResidual:output(zero),eulerRowIsProbability:le(zero,eulerOne)&&le(eulerOne,one)});
  power=mul(power,factor);
 }
 const current=history[history.length-1],q=exponential(mul(lambda,time)),pi0=pi===null?null:sub(one,pi);
 const kernel=pi===null?[[point(one),point(zero)],[point(zero),point(one)]]:[
  [shift(scale(q,pi),pi0!),scale(subtract(point(one),q),pi)],
  [scale(subtract(point(one),q),pi0!),shift(scale(q,pi0!),pi)],
 ];
 const eulerStochastic=le(mul(a,dt),one)&&le(mul(b,dt),one);
 return{parameters:p,precisionBits:256,rateUnits:'s^-1',timeUnits:'s',generator:generator.map(row=>row.map(output)),eulerStepKernel:eulerKernel.map(row=>row.map(output)),exactTimeKernel:kernel.map(row=>row.map(observation)),
  rates:[output(a),output(b)],relaxationRate:output(lambda),eulerStepTime:output(dt),eulerDeviationMultiplier:output(factor),
  stationaryDistribution:pi===null?null:[output(pi0!),output(pi)],selectedLimitDistribution:pi===null?[output(sub(one,u)),output(u)]:[output(pi0!),output(pi)],
  meanHoldingTimes:[a,b].map(rate=>rate.n===Z?{kind:'infinite' as const}:{kind:'finite' as const,value:output(div(one,rate))}),
  noJumpSurvival:[a,b].map(rate=>observation(exponential(mul(rate,time)))),history,current,
  classification:{stationaryIsUnique:lambda.n!==Z,irreducible:a.n>Z&&b.n>Z,stationaryFamily:lambda.n===Z?'all-initial-laws':'unique',
   initialLawIsStationary:lambda.n===Z||d.n===Z,currentExactLawIsStationary:lambda.n===Z||d.n===Z,
   generatorConservesMass:true,eulerKernelConservesMass:true,eulerKernelIsStochastic:eulerStochastic,
   eulerSelectedHistoryIsProbability:history.every(row=>row.eulerRowIsProbability),
   eulerLinearModeStrictlyContracts:lambda.n>Z&&le(abs(factor),one)&&abs(factor).n!==abs(factor).d,
   eulerLinearModeIsMarginal:abs(factor).n===abs(factor).d,eulerLinearModeExpands:!le(abs(factor),one)},
  approximationContract:'Two-state continuous-time row-law mu dot=mu Q. Nonnegative rates are exact rationals in s^-1 and bounded by5 per second for this interactive calculation; requested physical time0..10 seconds and exact initial occupancy0..1. Euler uses1..128 equal steps at the same final time. Directed256-bit exponential intervals describe the exact continuous-time law; rational Euler values are approximations and may be signed when I+dt Q is not stochastic. Do not clip or renormalize a failed Euler law. Linear-mode contraction, kernel positivity and a favorable stationary initial law are different properties. Zero rates preserve the entire stationary family. No thermal reservoir, heat, work or entropy production is inferred. No sampled trajectory or statistical fit is generated.'};
}
