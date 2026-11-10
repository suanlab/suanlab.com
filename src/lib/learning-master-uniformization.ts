/** Three-state continuous-time teaching model.
 * Exact conditional powers and positive Poisson tails; never normalize a truncation. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F} from '@/lib/learning-finite-markov-exact';
import {createTranscendentalContext} from '@/lib/learning-interval-transcendentals';
import {partitionObservationFromRange,type PartitionObservation as Observation} from '@/lib/learning-partition-exact-core';
export interface UniformizationParameters {
 clockwiseNumerator:number;counterclockwiseNumerator:number;rateDenominator:number;
 clockNumerator:number;clockDenominator:number;timeNumerator:number;timeDenominator:number;
 initialZeroWeight:number;initialOneWeight:number;initialTwoWeight:number;poissonCutoff:number;
}
interface I {lower:F;upper:F}
const Z=BigInt(0),O=BigInt(1),zero=f(Z),one=f(O);
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n);
const le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,min=(a:F,b:F)=>le(a,b)?a:b;
const sum=(xs:readonly F[])=>xs.reduce(add,zero),output=(a:F)=>exact(a.n,a.d),binary=(a:bigint)=>({coefficient:a,exponent:0});
const scale=(a:I,b:F):I=>({lower:mul(a.lower,b),upper:mul(a.upper,b)});
function observation(a:I):Observation {
 if(a.lower.n*a.upper.d===a.upper.n*a.lower.d)return output(a.lower);
 return partitionObservationFromRange({lower:{n:binary(a.lower.n),d:binary(a.lower.d)},upper:{n:binary(a.upper.n),d:binary(a.upper.d)}});
}
const integer=(x:unknown,min:number,max=1000000):x is number=>typeof x==='number'&&Number.isInteger(x)&&x>=min&&x<=max;
export function isUniformizationParameters(value:unknown):value is UniformizationParameters {
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const p=value as Record<string,unknown>,keys=['clockwiseNumerator','counterclockwiseNumerator','rateDenominator','clockNumerator','clockDenominator','timeNumerator','timeDenominator','initialZeroWeight','initialOneWeight','initialTwoWeight','poissonCutoff'];
 if(Object.keys(p).length!==keys.length||!keys.every(k=>Object.prototype.hasOwnProperty.call(p,k))||!keys.every(k=>integer(p[k],['rateDenominator','clockDenominator','timeDenominator'].includes(k)?1:0,k==='poissonCutoff'?256:1000000)))return false;
 const v=p as unknown as UniformizationParameters;
 return v.clockwiseNumerator<=5*v.rateDenominator&&v.counterclockwiseNumerator<=5*v.rateDenominator&&v.clockNumerator<=10*v.clockDenominator&&v.timeNumerator<=10*v.timeDenominator
 &&(v.clockwiseNumerator+v.counterclockwiseNumerator)*v.clockDenominator<=v.clockNumerator*v.rateDenominator&&v.initialZeroWeight+v.initialOneWeight+v.initialTwoWeight>0;
}
export function uniformizationModel(input:UniformizationParameters){
 if(!isUniformizationParameters(input))throw new RangeError('Eleven owned integer fields, nonzero initial weight sum, rates0..5/s, dominating clock0..10/s, time0..10s and Poisson cutoff0..256');
 const p=Object.freeze({...input}),a=f(BigInt(p.clockwiseNumerator),BigInt(p.rateDenominator)),b=f(BigInt(p.counterclockwiseNumerator),BigInt(p.rateDenominator)),rate=add(a,b),clock=f(BigInt(p.clockNumerator),BigInt(p.clockDenominator)),time=f(BigInt(p.timeNumerator),BigInt(p.timeDenominator)),mu=mul(clock,time),M=p.poissonCutoff;
 const total=BigInt(p.initialZeroWeight+p.initialOneWeight+p.initialTwoWeight),initial=[p.initialZeroWeight,p.initialOneWeight,p.initialTwoWeight].map(v=>f(BigInt(v),total));
 const generator=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>i===j?neg(rate):j===(i+1)%3?a:b));
 const identity=Array.from({length:3},(_,i)=>Array.from({length:3},(_,j)=>i===j?one:zero));
 const kernel=clock.n===Z?identity:generator.map((row,i)=>row.map((q,j)=>add(i===j?one:zero,div(q,clock))));
 const mixture=certifiedThreeStatePoissonMixture(kernel,initial,mu,M);
 const active=rate.n!==Z,selfLoop=clock.n===Z?one:sub(one,div(rate,clock));
 return{parameters:p,precisionBits:256,rateUnits:'s^-1',timeUnits:'s',generator:generator.map(row=>row.map(output)),potentialEventKernel:kernel.map(row=>row.map(output)),initialLaw:initial.map(output),exitRate:output(rate),potentialEventRate:output(clock),poissonMean:output(mu),potentialSelfLoopProbability:output(selfLoop),
  expectedPotentialEvents:output(mu),expectedActualStateChanges:output(mul(rate,time)),
  stationaryDistribution:active?[output(f(O,BigInt(3))),output(f(O,BigInt(3))),output(f(O,BigInt(3)))]:null,
  stationaryDirectedEdgeCurrent:active?output(div(sub(a,b),f(BigInt(3)))):null,
  ...mixture,
  classification:{generatorConservesMass:true,kernelIsStochastic:true,stationaryIsUnique:active,irreducible:active,stationaryFamily:active?'unique-uniform':'all-initial-laws',embeddedPeriod:!active?1:selfLoop.n!==Z||a.n!==Z&&b.n!==Z?1:3,exactLawConvergesToUniqueStationaryFromEveryInitialLaw:active,potentialEventsEqualStateChanges:clock.n===rate.n&&clock.d===rate.d,positiveOmittedMass:mu.n!==Z,truncatedMassIsOne:mu.n===Z},
  approximationContract:'Finite continuous-time three-cycle, row-law convention. Clockwise/counterclockwise rates are exact nonnegative rationals in s^-1, each at most5; dominating potential-event rate is at least their sum and at most10. Physical time0..10 seconds and cutoff0..256 are computation bounds. Conditional laws and bare Poisson coefficients are exact rationals. Directed256-bit exponential enclosures and a positive geometric suffix bound certify omitted Poisson mass; even a displayed partial mass of one can omit positive mass. Partial laws are not normalized. Component intervals enclose the complete semigroup law and can be unresolved when the cutoff is too small. Changing the dominating clock changes this representation and truncation, not the generator or exact physical law. The three homogeneous exit rates make the true jump count a thinned Poisson count; this assertion is not generalized to state-dependent holding rates. No thermal bath, heat or thermodynamic entropy production is specified.'};
}

/** Shared exact engine for a validated three-state potential-event chain.
 * The retained law stays a subprobability law; component enclosures can be broad. */
export function certifiedThreeStatePoissonMixture(kernel:readonly (readonly F[])[],initial:readonly F[],mu:F,M:number){
 const valid=(q:F)=>q&&typeof q.n==='bigint'&&typeof q.d==='bigint'&&q.d>Z&&q.n>=Z;
 if(!integer(M,0,256)||!valid(mu)||!le(mu,f(BigInt(100)))||initial.length!==3||!initial.every(valid)||sum(initial).n!==sum(initial).d||kernel.length!==3||kernel.some(row=>row.length!==3||!row.every(valid)||sum(row).n!==sum(row).d))throw new RangeError('Exact three-state stochastic kernel, initial law, Poisson mean0..100 and cutoff0..256 required');
 const ctx=createTranscendentalContext(256),e=ctx.expNegativeRatio(binary(mu.n),binary(mu.d)),exponential={lower:f(e.lower,ctx.scale),upper:f(e.upper,ctx.scale)};
 // c_k=mu^k/k! stays exact. A positive geometric remainder bounds the
 // infinite suffix after L, with a relative target tied to c_(M+1).
 const coefficients:F[]=[one];let remainder=zero,L=0;
 if(mu.n!==Z){
  const epsilon=f(O,O<<BigInt(256));
  for(let k=0;k<1024;k++){
   const next=div(mul(coefficients[k],mu),f(BigInt(k+1))),ratio=div(mu,f(BigInt(k+2)));
   if(k>=M+1&&le(ratio,one)&&ratio.n!==ratio.d){remainder=div(next,sub(one,ratio));if(le(remainder,mul(coefficients[M+1],epsilon))){L=k;break;}}
   coefficients.push(next);L=k+1;
   if(k===1023)throw new RangeError('Positive Poisson-tail bound did not converge within the declared computation limit');
  }
 }else{
  while(coefficients.length<=M+1)coefficients.push(zero);L=M+1;
 }
 const suffix:F[]=Array(L+2).fill(zero);for(let k=L;k>=0;k--)suffix[k]=add(coefficients[k],suffix[k+1]);
 let law=[...initial],weighted=[zero,zero,zero],bareMass=zero;const history=[];
 for(let k=0;k<=M;k++){
  const coefficient=coefficients[k],weight=scale(exponential,coefficient);bareMass=add(bareMass,coefficient);weighted=weighted.map((q,j)=>add(q,mul(coefficient,law[j])));
  const partial=weighted.map(q=>scale(exponential,q)),tail={lower:mul(exponential.lower,suffix[k+1]),upper:min(one,mul(exponential.upper,add(suffix[k+1],remainder)))};
  // The missing conditional law is a probability vector. This certifies
  // each component without turning the positive partial law into a normalized law.
  const certified=partial.map(q=>({lower:q.lower,upper:min(one,add(q.upper,tail.upper))}));
  history.push({cutoff:k,conditionalLaw:law.map(output),poissonWeight:observation(weight),partialLaw:partial.map(observation),partialMass:observation(scale(exponential,bareMass)),omittedPoissonMass:observation(tail),exactLawEnclosures:certified.map(observation),componentErrorUpperBound:output(tail.upper)});
  law=Array.from({length:3},(_,j)=>sum(law.map((q,i)=>mul(q,kernel[i][j]))));
 }
 return{history,current:history[history.length-1],tailProof:{lastSummedIndex:L,firstOmittedBareCoefficient:output(mu.n===Z?zero:div(mul(coefficients[L],mu),f(BigInt(L+1)))),bareRemainderUpperBound:output(remainder),relativeTargetBits:256}};
}
