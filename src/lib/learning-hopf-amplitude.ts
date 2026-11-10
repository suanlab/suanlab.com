/** Exact-sign supercritical Stuart–Landau preparation. Numerical foundation;
 * no reader registration, Worker or physical-system universality is asserted. */
import { fraction as f,zero,one,add,sub,mul,div,neg,equal,exact,logarithm,observeRange,entropyScale as S,type F } from './learning-entropy-production-math';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { directedPhaseRotation } from './learning-directed-phase-rotation';
import { isHopfAmplitudeParameters, hopfAmplitudeKeys, type HopfAmplitudeParameters } from './learning-hopf-amplitude-input';
export { isHopfAmplitudeParameters, hopfAmplitudeKeys } from './learning-hopf-amplitude-input';
export type { HopfAmplitudeParameters } from './learning-hopf-amplitude-input';
type R={lower:F;upper:F};
const point=(x:F):R=>({lower:x,upper:x}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
const plus=(a:R,b:R):R=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
const negative=(a:R):R=>({lower:neg(a.upper),upper:neg(a.lower)}),minus=(a:R,b:R)=>plus(a,negative(b));
const scale=(a:R,b:F):R=>b.n>=BigInt(0)?{lower:mul(a.lower,b),upper:mul(a.upper,b)}:{lower:mul(a.upper,b),upper:mul(a.lower,b)};
const product=(a:R,b:R):R=>{const ends=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];return{lower:ends.reduce((x,y)=>le(x,y)?x:y),upper:ends.reduce((x,y)=>le(x,y)?y:x)};};
function quotient(a:R,b:R):R {if(b.lower.n<=BigInt(0))throw new RangeError('Positive denominator certificate lost');return product(a,{lower:div(one,b.upper),upper:div(one,b.lower)});}
const observation=(x:R)=>observeRange(x.lower,x.upper);
const log=(x:R):R=>{if(x.lower.n<=BigInt(0))throw new RangeError('Positive log certificate lost');return{lower:f(logarithm(x.lower).lower,S),upper:f(logarithm(x.upper).upper,S)};};
function rootInteger(n:bigint){if(n<BigInt(0))throw new RangeError('Negative square root');if(n<BigInt(2))return n;let x=BigInt(1)<<BigInt(Math.ceil(n.toString(2).length/2));for(;;){const y=(x+n/x)/BigInt(2);if(y>=x)return x;x=y;}}
function sqrt(x:R):R {if(x.lower.n<BigInt(0))throw new RangeError('Nonnegative square-root certificate lost');const bound=(q:F,upper:boolean)=>{const nr=rootInteger(q.n),dr=rootInteger(q.d);if(nr*nr===q.n&&dr*dr===q.d)return f(nr,dr);const n=q.n*S*S,v=rootInteger(n/q.d);return f(v+(upper&&v*v*q.d!==n?BigInt(1):BigInt(0)),S);};return{lower:bound(x.lower,false),upper:bound(x.upper,true)};}
const square=(a:R):R=>{const a2=mul(a.lower,a.lower),b2=mul(a.upper,a.upper);return{lower:a.lower.n<=BigInt(0)&&a.upper.n>=BigInt(0)?zero:le(a2,b2)?a2:b2,upper:le(a2,b2)?b2:a2};};
const ctx=createTranscendentalContext(256),two=f(BigInt(2));
const decay=(x:F):R=>{if(x.n===BigInt(0))return point(one);const e=ctx.expNegativeRatio({coefficient:x.n,exponent:0},{coefficient:x.d,exponent:0});return{lower:f(e.lower,ctx.scale),upper:f(e.upper,ctx.scale)};};
function rotation(angle:R){
 const mid=div(add(angle.lower,angle.upper),two),radius=div(sub(angle.upper,angle.lower),two),v=directedPhaseRotation(mid.n,mid.d,256);
 const enclosure=(x:{lower:bigint;upper:bigint}):R=>{const lo=sub(f(x.lower,v.scale),radius),hi=add(f(x.upper,v.scale),radius);return{lower:le(lo,neg(one))?neg(one):lo,upper:le(one,hi)?one:hi};};
 return{cos:enclosure(v.cos),sin:enclosure(v.sin)};
}
export function hopfAmplitudeModel(input:HopfAmplitudeParameters){
 if(!isHopfAmplitudeParameters(input))throw new RangeError('Fifteen owned integer fields; magnitudes<=1e9, denominators/g positive, |mu|,g,|shear|,|omega|<=8,|x0|,|y0|<=4,time0..4s,records1..64');
 const parameters=Object.freeze(Object.fromEntries(hopfAmplitudeKeys.map(k=>[k,input[k]+0]))) as unknown as Readonly<HopfAmplitudeParameters>,p=parameters;
 const rational=(prefix:string)=>f(BigInt(p[(prefix+'Numerator') as keyof HopfAmplitudeParameters]),BigInt(p[(prefix+'Denominator') as keyof HopfAmplitudeParameters]));
 const mu=rational('mu'),g=rational('g'),c=rational('shear'),omega=rational('omega'),x0=rational('x'),y0=rational('y'),rho0=add(mul(x0,x0),mul(y0,y0)),T=rational('time'),origin=rho0.n===BigInt(0),positive=mu.n>BigInt(0),critical=mu.n===BigInt(0),rhoStar=positive?div(mu,g):zero;
 const stationaryAmplitude=origin||positive&&equal(rho0,rhoStar),circleFrequency=positive?sub(omega,mul(c,rhoStar)):null;
 const regime=critical?'critical-algebraic-attraction':positive?'unstable-origin-and-attracting-circle':'stable-origin';
 const history=Array.from({length:p.sampleSteps+1},(_,step)=>{
  const time=mul(T,f(BigInt(step),BigInt(p.sampleSteps))),e=decay(mul(two,mul(positive?mu:neg(mu),time)));
  let rho:R,integral:R,sensitivity:R;
  if(critical){const D=add(one,mul(two,mul(g,mul(rho0,time))));rho=point(div(rho0,D));integral=origin?point(zero):scale(log(point(D)),div(one,mul(two,g)));sensitivity=point(div(one,mul(D,D)));}
  else if(positive){const B=plus(e,scale(minus(point(one),e),div(mul(g,rho0),mu)));rho=quotient(point(rho0),B);integral=scale(plus(point(mul(two,mul(mu,time))),log(B)),div(one,mul(two,g)));sensitivity=quotient(e,square(B));}
  else{const D=plus(point(one),scale(minus(point(one),e),div(mul(g,rho0),neg(mu))));rho=quotient(scale(e,rho0),D);integral=scale(log(D),div(one,mul(two,g)));sensitivity=quotient(e,square(D));}
  // Structural identities avoid cancellation without changing any nonconstant flow.
  if(stationaryAmplitude){rho=point(rho0);integral=point(mul(rho0,time));}
  if(time.n===BigInt(0)){rho=point(rho0);integral=point(zero);sensitivity=point(one);}
  const phaseShift=origin?null:minus(point(mul(omega,time)),scale(integral,c)),frequency=minus(point(omega),scale(rho,c)),radialCoefficient=minus(point(mu),scale(rho,g));
  const gain=origin?null:sqrt(scale(rho,div(one,rho0))),turn=phaseShift?rotation(phaseShift):null;
  const x=turn&&gain?product(gain,minus(scale(turn.cos,x0),scale(turn.sin,y0))):point(zero),y=turn&&gain?product(gain,plus(scale(turn.sin,x0),scale(turn.cos,y0))):point(zero);
  const derivative=stationaryAmplitude?point(zero):scale(product(rho,radialCoefficient),two);
  const logBalance=origin?null:minus(log(scale(rho,div(one,rho0))),minus(point(mul(two,mul(mu,time))),scale(integral,mul(two,g))));
  return{step,time:exact(time),squaredAmplitude:observation(rho),amplitude:observation(sqrt(rho)),integratedSquaredAmplitude:observation(integral),phaseShift:phaseShift?observation(phaseShift):null,instantaneousAngularFrequency:origin?null:observation(frequency),x:observation(x),y:observation(y),squaredAmplitudeDerivative:observation(derivative),sensitivityToInitialSquaredAmplitude:observation(sensitivity),coordinateRadiusResidual:observation(minus(plus(square(x),square(y)),rho)),logAmplitudeBalanceResidual:logBalance?observation(logBalance):null,algebraicLogAmplitudeBalanceResidual:origin?null:exact(zero),phaseDefined:!origin};
 });
 return{parameters,precisionBits:256,initialSquaredAmplitude:exact(rho0),regime,
  classification:{positiveSaturation:true,originInitial:origin,stationaryAmplitude,criticalParameterExactlyZero:critical,ordinaryHopfCrossingAtMuZero:omega.n!==BigInt(0),positiveMuCircleExists:positive,positiveMuCircleIsPeriodicOrbit:positive&&circleFrequency!.n!==BigInt(0),positiveMuCircleIsEquilibria:positive&&circleFrequency!.n===BigInt(0),phaseIsDefined:!origin,sampledNoise:false},
  originLinearAmplitudeGrowthRate:exact(mu),originLinearSquaredAmplitudeGrowthRate:exact(mul(two,mu)),
  attractingCircleSquaredAmplitude:positive?exact(rhoStar):null,attractingCircleAngularFrequency:circleFrequency?exact(circleFrequency):null,
  attractingCircleRadialDecayRate:positive?exact(mul(two,mu)):null,
  originLinearAmplitudeRelaxationTime:mu.n<BigInt(0)?exact(div(one,neg(mu))):null,
  attractingCircleRadialRelaxationTime:positive?exact(div(one,mul(two,mu))):null,
  criticalSquaredAmplitudeHalfTime:critical&&!origin?exact(div(one,mul(two,mul(g,rho0)))):null,
  criticalAmplitudeHalfTime:critical&&!origin?exact(div(f(BigInt(3)),mul(two,mul(g,rho0)))):null,
  periodDividedByTwoPi:positive&&circleFrequency!.n!==BigInt(0)?exact(div(one,circleFrequency!.n>BigInt(0)?circleFrequency!:neg(circleFrequency!))):null,
  history,current:history.at(-1)!,
  approximationContract:'Deterministic supercritical cubic Stuart-Landau vector field dz/dt=(mu+i omega)z-(g+i c)|z|^2 z, g>0; amplitudes dimensionless, coefficients in inverse seconds. Exact rational parameter sign/equality and critical/stationary-amplitude branches. Squared-amplitude Riccati flow and integrated amplitude use directed256-bit exp/log endpoints with exact rational scaling; sqrt and angle-halving sin/cos have outward bounds. This is an analytic model flow, not a time-step integrator or sampled noise. Phase is undefined at the exact origin. A positive-mu invariant circle is an equilibrium circle if its angular speed is zero; it is not then a periodic orbit. Ordinary Hopf at mu0 additionally requires omega nonzero; global use of the polynomial field is not a universality claim for physical systems. Time-record count affects display only. Coordinate/radius and log-balance residuals are independently enclosed, and algebraic identities are separate. No reader, Worker, empirical inference or whole-topic completion claim.',
 };
}
