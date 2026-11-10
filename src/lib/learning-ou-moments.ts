/** Whole-line OU moments. Exact rational EM moments and directed
 * exponential enclosures; no density mesh, trajectory sampler or UI worker. */
import {markovFraction as f, exactMarkovObservation as exact, type MarkovFraction as F} from '@/lib/learning-finite-markov-exact';
import {createTranscendentalContext} from '@/lib/learning-interval-transcendentals';
import {partitionObservationFromRange, type PartitionObservation as Observation} from '@/lib/learning-partition-exact-core';
export interface OUMomentParameters {
 relaxationNumerator:number; diffusionNumerator:number; coefficientDenominator:number;
 initialMeanNumerator:number; initialVarianceNumerator:number; initialDenominator:number;
 timeNumerator:number; timeDenominator:number; eulerSteps:number;
}
interface Interval {lower:F;upper:F}
const Z=BigInt(0),O=BigInt(1),TWO=BigInt(2);
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d);
const neg=(a:F)=>f(-a.n,a.d);
const sub=(a:F,b:F)=>add(a,neg(b));
const mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d);
const div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n);
const equal=(a:F,b:F)=>a.n*b.d===b.n*a.d;
const point=(a:F):Interval=>({lower:a,upper:a});
const scale=(a:Interval,b:F):Interval=>b.n<Z?{lower:mul(a.upper,b),upper:mul(a.lower,b)}:{lower:mul(a.lower,b),upper:mul(a.upper,b)};
const sum=(a:Interval,b:Interval):Interval=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
const difference=(a:Interval,b:Interval)=>sum(a,scale(b,f(-O)));
const output=(a:F)=>exact(a.n,a.d);
const binary=(coefficient:bigint)=>({coefficient,exponent:0});
function observe(a:Interval):Observation {
 if(equal(a.lower,a.upper))return output(a.lower);
 return partitionObservationFromRange({lower:{n:binary(a.lower.n),d:binary(a.lower.d)},upper:{n:binary(a.upper.n),d:binary(a.upper.d)}});
}
const integer=(x:unknown,min:number,max=1000000):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
export function isOUMomentParameters(value:unknown):value is OUMomentParameters {
 if(!value||typeof value!=='object'||Array.isArray(value))return false;
 const p=value as Record<string,unknown>;
 const fields=['relaxationNumerator','diffusionNumerator','coefficientDenominator','initialMeanNumerator','initialVarianceNumerator','initialDenominator','timeNumerator','timeDenominator','eulerSteps'];
 return Object.keys(p).length===fields.length&&fields.every(k=>Object.prototype.hasOwnProperty.call(p,k))
 &&['relaxationNumerator','diffusionNumerator','initialVarianceNumerator','timeNumerator'].every(k=>integer(p[k],0))
 &&['coefficientDenominator','initialDenominator','timeDenominator'].every(k=>integer(p[k],1))
 &&integer(p.initialMeanNumerator,-1000000)&&integer(p.eulerSteps,1,128)
 &&(p.relaxationNumerator as number)<=5*(p.coefficientDenominator as number)
 &&(p.diffusionNumerator as number)<=2*(p.coefficientDenominator as number)
 &&Math.abs(p.initialMeanNumerator as number)<=5*(p.initialDenominator as number)
 &&(p.initialVarianceNumerator as number)<=5*(p.initialDenominator as number)
 &&(p.timeNumerator as number)<=10*(p.timeDenominator as number);
}
export function ouMomentModel(input:OUMomentParameters){
 if(!isOUMomentParameters(input))throw new RangeError('Nine own integer fields; gamma0..5/s, D0..2m²/s, mean−5..5m, variance0..5m², time0..10s, steps1..128; denominators1..1000000');
 const p=Object.freeze({...input}),q=(n:number,d:number)=>f(BigInt(n),BigInt(d));
 const gamma=q(p.relaxationNumerator,p.coefficientDenominator),D=q(p.diffusionNumerator,p.coefficientDenominator);
 const m0=q(p.initialMeanNumerator,p.initialDenominator),v0=q(p.initialVarianceNumerator,p.initialDenominator),time=q(p.timeNumerator,p.timeDenominator);
 const one=f(O),zero=f(Z),two=f(TWO),dt=div(time,f(BigInt(p.eulerSteps))),factor=sub(one,mul(gamma,dt));
 const noise=mul(mul(two,D),dt),factorSquared=mul(factor,factor),ctx=createTranscendentalContext(256);
 const exponential=(a:F):Interval=>{const e=ctx.expNegativeRatio(binary(a.n),binary(a.d));return{lower:f(e.lower,ctx.scale),upper:f(e.upper,ctx.scale)}};
 const stationaryVariance=gamma.n===Z?null:div(D,gamma);
 const initialStationary=gamma.n===Z?D.n===Z:m0.n===Z&&equal(v0,stationaryVariance!);
 const stationaryVarianceDeviation=stationaryVariance===null?null:sub(v0,stationaryVariance);
 let emMean=m0,emVariance=v0;
 const history=[];
 for(let step=0;step<=p.eulerSteps;step++){
  const t=mul(dt,f(BigInt(step))),r=exponential(mul(gamma,t)),r2=exponential(mul(mul(two,gamma),t));
  const mean=scale(r,m0);
  // A sum of nonnegative terms bounds variance without a negative cancellation artifact.
  // Exact stationarity and Brownian limits are handled symbolically first.
  const variance=stationaryVariance===null?point(add(v0,mul(mul(two,D),t)))
   :equal(v0,stationaryVariance)?point(v0):sum(scale(r2,v0),scale(difference(point(one),r2),stationaryVariance));
  const varianceDeviation=stationaryVarianceDeviation===null?null:scale(r2,stationaryVarianceDeviation);
  const varianceRate=varianceDeviation===null?point(mul(two,D)):scale(varianceDeviation,neg(mul(two,gamma)));
  const atom=v0.n===Z&&(D.n===Z||t.n===Z);
  history.push({step,time:output(t),mean:observe(mean),variance:observe(variance),emMean:output(emMean),emVariance:output(emVariance),
   signedMeanError:observe(difference(point(emMean),mean)),signedVarianceError:observe(difference(point(emVariance),variance)),
   meanRate:observe(scale(mean,neg(gamma))),varianceRate:observe(varianceRate),
   signedVarianceDeviationFromStationary:varianceDeviation===null?null:observe(varianceDeviation),
   initialTimeCovariance:observe(scale(r,v0)),
   // For a stationary OU initialization only: two models with equal marginals but different joint laws.
   stationarySDETwoTimeCovariance:initialStationary&&stationaryVariance!==null?observe(scale(r,stationaryVariance)):null,
   stationaryFrozenODETwoTimeCovariance:initialStationary&&stationaryVariance!==null?output(stationaryVariance):null,
   lawKind:atom?'point-mass' as const:'ordinary-gaussian-density' as const,scoreDefined:!atom,
   normalizationResidual:output(zero),emLawIsProbability:true});
  emMean=mul(factor,emMean);emVariance=add(mul(factorSquared,emVariance),noise);
 }
 const contracting=factor.n*factor.n<factor.d*factor.d;
 return{parameters:p,precisionBits:256,units:{position:'m',time:'s',gamma:'s^-1',diffusion:'m^2/s',variance:'m^2'},
  relaxation:output(gamma),diffusion:output(D),initialMean:output(m0),initialVariance:output(v0),
  stepTime:output(dt),emMeanMultiplier:output(factor),emNoiseVariancePerStep:output(noise),
  stationaryVariance:stationaryVariance===null?null:output(stationaryVariance),
  emInvariantVariance:contracting?output(div(noise,sub(one,factorSquared))):null,
  classification:{initialLawIsStationary:initialStationary,stationaryProbability:gamma.n>Z?'unique':D.n>Z?'none-on-real-line':'all-initial-laws',
   stationaryLawKind:gamma.n===Z?null:D.n===Z?'point-mass':'ordinary-gaussian-density',
   emStrictMeanContraction:contracting,emMarginalMeanMode:factor.n*factor.n===factor.d*factor.d,
   emExpandingMeanMode:factor.n*factor.n>factor.d*factor.d,emAlwaysValidGaussianOrAtom:true,
   emInvariantVarianceIsUnique:contracting,
   probabilityFlowDefinedAtInitialTime:v0.n>Z,
   positiveTimeRegularizesAtom:D.n>Z&&v0.n===Z},history,current:history[history.length-1],
  approximationContract:'Whole-real-line Itô OU Gaussian/atom preparation: dX=−gamma X dt+sqrt(2D)dW. Independent initial Gaussian/atom, nonnegative constant gamma,D. Directed256-bit exponential intervals and exact rational Euler–Maruyama moments. Time and step count are distinct. EM laws remain probabilities even when their mean expands. An atom has no ordinary density or score. No density mesh, sampled trajectory, finite-volume boundary, thermal bath, fitted score or reverse-time sampler is implemented. Equal one-time marginals do not imply equal joint path laws. Native worker consumers distinguish display projections from exact bounds.'};
}
