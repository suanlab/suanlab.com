/** Stationary R² OU current family, exact rational physical
 * inputs, directed pi/exp/log/rotation and exact Euler covariance recurrence. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F}from'@/lib/learning-finite-markov-exact';
import {createTranscendentalContext}from'@/lib/learning-interval-transcendentals';
import {directedPi}from'@/lib/learning-directed-pi';
import {partitionObservationFromRange}from'@/lib/learning-partition-exact-core';
import {directedRationalRotation}from'./learning-directed-rational-rotation';
export interface CirculatingGaussianParameters{
 relaxationNumerator:number;diffusionNumerator:number;coefficientDenominator:number;
 angularNumerator:number;angularDenominator:number;
 probeXNumerator:number;probeYNumerator:number;spaceDenominator:number;
 timeNumerator:number;timeDenominator:number;eulerSteps:number;
}
const Z=BigInt(0),O=BigInt(1),zero=f(Z),one=f(O),two=f(BigInt(2));
const add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),abs=(a:F)=>a.n<Z?neg(a):a,le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
type I={lower:F;upper:F};const point=(a:F):I=>({lower:a,upper:a});
const shift=(a:I,b:F):I=>({lower:add(a.lower,b),upper:add(a.upper,b)});
const scale=(a:I,b:F):I=>b.n<Z?{lower:mul(a.upper,b),upper:mul(a.lower,b)}:{lower:mul(a.lower,b),upper:mul(a.upper,b)};
const difference=(a:I,b:I):I=>({lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)});
const product=(a:I,b:I):I=>{const q=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];return{lower:q.reduce((a,b)=>le(a,b)?a:b),upper:q.reduce((a,b)=>le(a,b)?b:a)}};
const positiveRatio=(a:I,b:I):I=>{if(b.lower.n<=Z)throw new RangeError('Positive interval denominator required');return product(a,{lower:div(one,b.upper),upper:div(one,b.lower)})};
const output=(a:F)=>exact(a.n,a.d),binary=(coefficient:bigint)=>({coefficient,exponent:0});
function observe(a:I){if(a.lower.n*a.upper.d===a.upper.n*a.lower.d)return output(a.lower);return partitionObservationFromRange({lower:{n:binary(a.lower.n),d:binary(a.lower.d)},upper:{n:binary(a.upper.n),d:binary(a.upper.d)}});}
const integer=(v:unknown,min:number,max=1000000):v is number=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=min&&v<=max;
export function isCirculatingGaussianParameters(value:unknown):value is CirculatingGaussianParameters{
 if(!value||typeof value!=='object'||Array.isArray(value))return false;const p=value as Record<string,unknown>;
 const fields=['relaxationNumerator','diffusionNumerator','coefficientDenominator','angularNumerator','angularDenominator','probeXNumerator','probeYNumerator','spaceDenominator','timeNumerator','timeDenominator','eulerSteps'];
 return Object.keys(p).length===fields.length&&fields.every(k=>Object.prototype.hasOwnProperty.call(p,k))
 &&['relaxationNumerator','diffusionNumerator','coefficientDenominator','angularDenominator','spaceDenominator','timeDenominator'].every(k=>integer(p[k],1))
 &&['angularNumerator','probeXNumerator','probeYNumerator'].every(k=>integer(p[k],-1000000))&&integer(p.timeNumerator,0)&&integer(p.eulerSteps,1,64)
 &&(p.relaxationNumerator as number)<=2*(p.coefficientDenominator as number)&&(p.diffusionNumerator as number)<=2*(p.coefficientDenominator as number)
 &&Math.abs(p.angularNumerator as number)<=4*(p.angularDenominator as number)
 &&Math.abs(p.probeXNumerator as number)<=4*(p.spaceDenominator as number)&&Math.abs(p.probeYNumerator as number)<=4*(p.spaceDenominator as number)
 &&(p.timeNumerator as number)<=4*(p.timeDenominator as number);
}
export function circulatingGaussianModel(input:CirculatingGaussianParameters){
 if(!isCirculatingGaussianParameters(input))throw new RangeError('Eleven own integer fields: gamma,D>0 up to2; angular speed−4..4rad/s, probe−4..4m, time0..4s, steps1..64; denominators1..1000000');
 const p=Object.freeze({...input}),q=(n:number,d:number)=>f(BigInt(n),BigInt(d)),gamma=q(p.relaxationNumerator,p.coefficientDenominator),D=q(p.diffusionNumerator,p.coefficientDenominator),omega=q(p.angularNumerator,p.angularDenominator),x=q(p.probeXNumerator,p.spaceDenominator),y=q(p.probeYNumerator,p.spaceDenominator),T=q(p.timeNumerator,p.timeDenominator),h=div(T,f(BigInt(p.eulerSteps))),variance=div(D,gamma),radiusSquared=add(mul(x,x),mul(y,y));
 const ctx=createTranscendentalContext(256),exp=(a:F):I=>{const e=ctx.expNegativeRatio(binary(a.n),binary(a.d));return{lower:f(e.lower,ctx.scale),upper:f(e.upper,ctx.scale)}},log=(a:F):I=>{const e=ctx.logRatio(binary(a.n),binary(a.d));return{lower:f(e.lower,ctx.scale),upper:f(e.upper,ctx.scale)}};
 const pp=directedPi(256),pi={lower:f(pp.lower,ctx.scale),upper:f(pp.upper,ctx.scale)},twopi=scale(pi,two),z=div(radiusSquared,mul(two,variance)),attenuation=exp(z);
 const density=positiveRatio(attenuation,scale(twopi,variance));
 const logpi={lower:log(pi.lower).lower,upper:log(pi.upper).upper},logDensity=difference(shift(log(div(one,mul(two,variance))),neg(z)),logpi);
 const logSigned=(coefficient:F)=>coefficient.n===Z?null:observe({lower:add(logDensity.lower,log(abs(coefficient)).lower),upper:add(logDensity.upper,log(abs(coefficient)).upper)});
 const drift=[sub(neg(mul(gamma,x)),mul(omega,y)),add(neg(mul(gamma,y)),mul(omega,x))],diffusive=[mul(gamma,x),mul(gamma,y)],currentVelocity=[neg(mul(omega,y)),mul(omega,x)];
 const a=sub(one,mul(gamma,h)),b=mul(omega,h),normSquared=add(mul(a,a),mul(b,b)),strict=normSquared.n<normSquared.d;
 const emNoise=mul(mul(two,D),h),invariant=strict?div(emNoise,sub(one,normSquared)):null;
 const covariance=(c:I,s:I,amplitude:F)=>[[observe(scale(c,amplitude)),observe(scale(s,neg(amplitude)))],[observe(scale(s,amplitude)),observe(scale(c,amplitude))]];
 let emVariance=variance,re=one,im=zero;const history=[];
 for(let step=0;step<=p.eulerSteps;step++){
  const t=mul(h,f(BigInt(step))),angle=mul(omega,t),rr=directedRationalRotation(angle.n,angle.d),sin={lower:f(rr.sin.lower,rr.scale),upper:f(rr.sin.upper,rr.scale)},cos={lower:f(rr.cos.lower,rr.scale),upper:f(rr.cos.upper,rr.scale)},decay=exp(mul(gamma,t));
  history.push({step,time:output(t),angleRadians:output(angle),exactCoordinateVariance:output(variance),eulerCoordinateVariance:output(emVariance),signedEulerVarianceError:output(sub(emVariance,variance)),
   sine:observe(sin),cosine:observe(cos),relaxationEnvelope:observe(decay),
   sdeCovXtX0:covariance(product(cos,decay),product(sin,decay),variance),probabilityFlowCovXtX0:covariance(cos,sin,variance),
   eulerCovXtX0:covariance(point(re),point(im),variance),mean:[output(zero),output(zero)],normalizationResidual:output(zero),eulerLawIsProbability:true});
  emVariance=add(mul(normSquared,emVariance),emNoise);const nextRe=sub(mul(a,re),mul(b,im));im=add(mul(b,re),mul(a,im));re=nextRe;
 }
 const circCoefficient=div(mul(mul(gamma,omega),radiusSquared),D);
 return{parameters:p,precisionBits:256,stationaryCoordinateVariance:output(variance),probeRadiusSquared:output(radiusSquared),density:observe(density),logDensity:observe(logDensity),
  driftVelocity:drift.map(output),diffusiveProbabilityVelocity:diffusive.map(output),probabilityFlowVelocity:currentVelocity.map(output),
  driftCurrent:drift.map(v=>observe(scale(density,v))),diffusiveCurrent:diffusive.map(v=>observe(scale(density,v))),totalCurrent:currentVelocity.map(v=>observe(scale(density,v))),logAbsoluteCurrent:currentVelocity.map(logSigned),
  stationaryCurrentDivergence:output(zero),circleNormalProbabilityFlux:output(zero),
  velocityCirculation:observe(scale(twopi,mul(omega,radiusSquared))),currentTangentialLineIntegral:observe(scale(attenuation,circCoefficient)),
  stepTime:output(h),eulerAmplificationSquared:output(normSquared),eulerNoiseVariance:output(emNoise),eulerInvariantCoordinateVariance:invariant===null?null:output(invariant),
  linearObservableAsymptoticVariance:output(div(mul(two,D),add(mul(gamma,gamma),mul(omega,omega)))),radialObservableAsymptoticVariance:output(div(mul(f(BigInt(4)),mul(D,D)),mul(mul(gamma,gamma),gamma))),
  linearAsymptoticVarianceRatioToZeroRotation:output(div(mul(gamma,gamma),add(mul(gamma,gamma),mul(omega,omega)))),
  classification:{densityStrictlyPositive:true,initialLawIsStationary:true,coordinatesAreEvenPositions:true,probeCurrentIsExactlyZero:omega.n===Z||radiusSquared.n===Z,
   ordinaryStationaryCurrentVanishesEverywhere:omega.n===Z,eulerStrictMeanContraction:strict,eulerMarginalMeanMode:normSquared.n===normSquared.d,eulerExpandingMeanMode:normSquared.n>normSquared.d,
   eulerGaussianLawsRemainProbabilities:true,radialAsymptoticVarianceChangesWithRotation:false,linearDriftRealDecayRateIndependentOfRotation:true},history,current:history[history.length-1],
  approximationContract:'Stationary whole-plane Gaussian OU with gamma,D>0, A=−gamma I+omega R and R=[[0,−1],[1,0]]. Exact rational radians, probe coordinates and Euler covariance, directed256bit pi/exp/log/sin/cos and interval products. Density is positive analytically even when a linear enclosure is unresolved; inspect its separate logarithm. Current/velocity circulation is a tangential line integral, not outward mass crossing a circle. Covariance order is E[Xt X0^T]. Equal marginal laws do not imply equal path laws. Asymptotic variances concern specified stationary time-average observables; no universal mixing-rate improvement, thermal bath or entropy-production inference. Exact2pi sampling aliases are analytic examples, not rounded rational-radian preset identities. Native worker consumers validate preparation identity and distinguish display projections from exact bounds.'};
}
