/** Clockwise cycle with state-dependent holding rates.
 * Jump-instants and physical-time occupations use different probability laws. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F} from '@/lib/learning-finite-markov-exact';
import {createTranscendentalContext} from '@/lib/learning-interval-transcendentals';
import {partitionObservationFromRange} from '@/lib/learning-partition-exact-core';
import {certifiedThreeStatePoissonMixture} from './learning-master-uniformization';
export interface HoldingParameters {
 rateZeroNumerator:number;rateOneNumerator:number;rateTwoNumerator:number;rateDenominator:number;
 clockNumerator:number;clockDenominator:number;timeNumerator:number;timeDenominator:number;
 initialZeroWeight:number;initialOneWeight:number;initialTwoWeight:number;poissonCutoff:number;
}
const Z=BigInt(0),O=BigInt(1),zero=f(Z),one=f(O);
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),sub=(a:F,b:F)=>f(a.n*b.d-b.n*a.d,a.d*b.d),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),sum=(xs:readonly F[])=>xs.reduce(add,zero),output=(q:F)=>exact(q.n,q.d);
const integer=(x:unknown,min:number,max=1000000):x is number=>typeof x==='number'&&Number.isInteger(x)&&x>=min&&x<=max;
export function isHoldingParameters(value:unknown):value is HoldingParameters {
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const p=value as Record<string,unknown>,keys=['rateZeroNumerator','rateOneNumerator','rateTwoNumerator','rateDenominator','clockNumerator','clockDenominator','timeNumerator','timeDenominator','initialZeroWeight','initialOneWeight','initialTwoWeight','poissonCutoff'];
 if(Object.keys(p).length!==keys.length||!keys.every(k=>Object.prototype.hasOwnProperty.call(p,k))||!keys.every(k=>integer(p[k],['rateDenominator','clockDenominator','timeDenominator'].includes(k)?1:0,k==='poissonCutoff'?256:1000000)))return false;
 const v=p as unknown as HoldingParameters,rates=[v.rateZeroNumerator,v.rateOneNumerator,v.rateTwoNumerator];return rates.every(n=>n<=5*v.rateDenominator&&n*v.clockDenominator<=v.clockNumerator*v.rateDenominator)&&v.clockNumerator<=10*v.clockDenominator&&v.timeNumerator<=10*v.timeDenominator&&v.initialZeroWeight+v.initialOneWeight+v.initialTwoWeight>0;
}
export function holdingModel(input:HoldingParameters){
 if(!isHoldingParameters(input))throw new RangeError('Twelve owned integer fields; nonzero initial weights; exit rates0..5/s, dominating clock0..10/s, time0..10s, cutoff0..256');
 const p=Object.freeze({...input}),rates=[p.rateZeroNumerator,p.rateOneNumerator,p.rateTwoNumerator].map(n=>f(BigInt(n),BigInt(p.rateDenominator))),clock=f(BigInt(p.clockNumerator),BigInt(p.clockDenominator)),time=f(BigInt(p.timeNumerator),BigInt(p.timeDenominator));
 const weights=[p.initialZeroWeight,p.initialOneWeight,p.initialTwoWeight],total=BigInt(weights.reduce((a,b)=>a+b,0)),initial=weights.map(n=>f(BigInt(n),total));
 const generator=rates.map((r,i)=>Array.from({length:3},(_,j)=>i===j?f(-r.n,r.d):j===(i+1)%3?r:zero));
 const potential=generator.map((row,i)=>row.map((q,j)=>clock.n===Z?(i===j?one:zero):add(i===j?one:zero,div(q,clock))));
 const embedded=rates.map((rate,i)=>rate.n===Z?null:Array.from({length:3},(_,j)=>output(j===(i+1)%3?one:zero)));
 const absorbing=rates.flatMap((q,i)=>q.n===Z?[i]:[]),allPositive=absorbing.length===0,unique=allPositive||absorbing.length===1;
 const holding=rates.map(q=>q.n===Z?null:div(one,q)),cycleTime=allPositive?sum(holding as F[]):null;
 const pi=allPositive?(holding as F[]).map(q=>div(q,cycleTime!)):absorbing.length===1?rates.map((_,i)=>i===absorbing[0]?one:zero):null;
 const absorptionTarget=allPositive?null:rates.map((_,start)=>{for(let k=0;k<3;k++){const target=(start+k)%3;if(rates[target].n===Z)return target;}throw new Error('Absorbing target must exist');});
 const selectedLimit=allPositive?pi!:rates.map((_,target)=>sum(initial.filter((_,i)=>absorptionTarget![i]===target)));
 const selectedAbsorptionMean=allPositive?null:sum(initial.map((w,start)=>{let t=zero;for(let k=0;k<3;k++){const i=(start+k)%3;if(rates[i].n===Z)break;t=add(t,holding[i]!);}return mul(w,t);}));
 const departureRate=allPositive?div(f(BigInt(3)),cycleTime!):zero;
 const mixture=certifiedThreeStatePoissonMixture(potential,initial,mul(clock,time),p.poissonCutoff),ctx=createTranscendentalContext(256);
 const survival=rates.map(rate=>{const q=mul(rate,time);if(q.n===Z)return output(one);const e=ctx.expNegativeRatio({coefficient:q.n,exponent:0},{coefficient:q.d,exponent:0});return partitionObservationFromRange({lower:{n:{coefficient:e.lower,exponent:0},d:{coefficient:ctx.scale,exponent:0}},upper:{n:{coefficient:e.upper,exponent:0},d:{coefficient:ctx.scale,exponent:0}}});});
 const initialIsStationary=pi?initial.every((q,i)=>q.n*pi[i].d===pi[i].n*q.d):initial.every((q,i)=>q.n===Z||rates[i].n===Z);
 return{parameters:p,precisionBits:256,rateUnits:'s^-1',timeUnits:'s',exitRates:rates.map(output),generator:generator.map(row=>row.map(output)),potentialEventKernel:potential.map(row=>row.map(output)),actualJumpKernelRows:embedded,initialLaw:initial.map(output),
 meanHoldingTimes:holding.map(q=>q?{kind:'finite' as const,value:output(q)}:{kind:'infinite' as const}),noActualJumpSurvival:survival,
 stationaryDistribution:pi?.map(output)??null,stationaryFamilyAbsorbingStates:absorbing,selectedLimitDistribution:selectedLimit.map(output),
 longRunActualJumpVisitLaw:allPositive?[output(f(O,BigInt(3))),output(f(O,BigInt(3))),output(f(O,BigInt(3)))]:null,
 meanCompleteCycleTime:cycleTime?output(cycleTime):null,stationaryMeanJumpRate:output(departureRate),stationaryPerEdgeCurrent:output(allPositive?div(one,cycleTime!):zero),
 selectedAbsorptionTargets:absorptionTarget,selectedMeanAbsorptionTime:selectedAbsorptionMean===null?null:output(selectedAbsorptionMean),
 potentialSelfLoopProbabilities:rates.map(rate=>output(clock.n===Z?one:sub(one,div(rate,clock)))),...mixture,
 classification:{irreducible:allPositive,stationaryIsUnique:unique,stationaryFamily:allPositive?'unique-holding-weighted':unique?'unique-absorbing':'simplex-on-absorbing-states',initialLawIsStationary:initialIsStationary,actualJumpSequenceTerminatesAlmostSurely:!allPositive,actualJumpKernelDefinedOnEveryState:allPositive,actualJumpChainPeriod:allPositive?3:null,convergesToUniqueStationaryFromEveryInitialLaw:unique,selectedLawHasLimit:true,actualCountHasOneHomogeneousPoissonLawForAllInitialStates:rates.every(q=>q.n*rates[0].d===rates[0].n*q.d),positiveOmittedMass:clock.n!==Z&&time.n!==Z},
 approximationContract:'State-dependent clockwise continuous-time three-cycle. Nonnegative exit rates0..5/s have common exact denominator; a dominating potential-event clock0..10/s and physical time0..10s feed the shared positive Poisson-mixture certificate at cutoff0..256. These are interaction bounds. With all exit rates positive, the long-run physical-time law is proportional to inverse exit rates, while actual-jump visit frequencies are uniform and the period-three jump laws need not converge. A zero exit rate is absorbing: the actual next-jump row is undefined, not a fabricated self-jump. Stationary families are supported on zero-rate states and the selected physical-time limit is determined by initial absorption basins. Waiting times are exponential; their reciprocal rates are means, not deterministic delays. For unequal exit rates, no single homogeneous Poisson law for actual jump counts is asserted. No heat bath, detailed thermodynamic balance or entropy production is supplied.'};
}
