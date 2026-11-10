/** Explicit thermal OU family: D=mu*theta, gamma=mu*kappa.
 * Directed analytic response/correlation matrices, not sampled trajectories. */
import { markovFraction as fraction, exactMarkovObservation as exact, type MarkovFraction as F } from '@/lib/learning-finite-markov-exact';
import { createTranscendentalContext } from '@/lib/learning-interval-transcendentals';
import { directedRationalRotation } from '@/lib/learning-directed-rational-rotation';
import { partitionObservationFromRange } from '@/lib/learning-partition-exact-core';

export interface ThermalOuResponseParameters {
  mobilityNumerator: number; stiffnessNumerator: number; thermalEnergyNumerator: number; coefficientDenominator: number;
  angularNumerator: number; angularDenominator: number;
  forceXNumerator: number; forceYNumerator: number; forceDenominator: number;
  impulseXNumerator: number; impulseYNumerator: number; impulseDenominator: number;
  timeNumerator: number; timeDenominator: number; sampleSteps: number;
}
const Z=BigInt(0),O=BigInt(1),zero=fraction(Z),one=fraction(O);
const add=(a:F,b:F)=>fraction(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>fraction(-a.n,a.d);
const mul=(a:F,b:F)=>fraction(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>fraction(a.n*b.d,a.d*b.n);
const le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
interface I { lower:F; upper:F }
const point=(a:F):I=>({lower:a,upper:a});
const plus=(a:I,b:I):I=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
const scale=(a:I,b:F):I=>b.n<Z?{lower:mul(a.upper,b),upper:mul(a.lower,b)}:{lower:mul(a.lower,b),upper:mul(a.upper,b)};
const minus=(a:I,b:I):I=>plus(a,scale(b,fraction(-O)));
function product(a:I,b:I):I {
  const ends=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];
  return {lower:ends.reduce((a,b)=>le(a,b)?a:b),upper:ends.reduce((a,b)=>le(a,b)?b:a)};
}
const containsZero=(a:I)=>a.lower.n<=Z&&a.upper.n>=Z;
function ratio(a:I,b:I):I {
  if(containsZero(b))throw new RangeError('Response denominator enclosure includes zero');
  return product(a,{lower:div(one,b.upper),upper:div(one,b.lower)});
}
const observe=(a:I)=>a.lower.n*a.upper.d===a.upper.n*a.lower.d?exact(a.lower.n,a.lower.d):partitionObservationFromRange({
  lower:{n:{coefficient:a.lower.n,exponent:0},d:{coefficient:a.lower.d,exponent:0}},
  upper:{n:{coefficient:a.upper.n,exponent:0},d:{coefficient:a.upper.d,exponent:0}},
});
const output=(a:F)=>exact(a.n,a.d);
const matrix=(c:I,s:I)=>[[observe(c),observe(scale(s,fraction(-O)))],[observe(s),observe(c)]];
const vector=(c:I,s:I,v:F[])=>[observe(minus(scale(c,v[0]),scale(s,v[1]))),observe(plus(scale(s,v[0]),scale(c,v[1])))];
const integer=(v:unknown,min:number,max=1000000):v is number=>typeof v==='number'&&Number.isSafeInteger(v)&&v>=min&&v<=max;
const keys=['mobilityNumerator','stiffnessNumerator','thermalEnergyNumerator','coefficientDenominator','angularNumerator','angularDenominator','forceXNumerator','forceYNumerator','forceDenominator','impulseXNumerator','impulseYNumerator','impulseDenominator','timeNumerator','timeDenominator','sampleSteps'] as const;
export function isThermalOuResponseParameters(value:unknown):value is ThermalOuResponseParameters {
  if(!value||typeof value!=='object'||Object.getPrototypeOf(value)!==Object.prototype)return false;
  const p=value as Record<string,number>,signed=['angularNumerator','forceXNumerator','forceYNumerator','impulseXNumerator','impulseYNumerator'];
  return Reflect.ownKeys(p).length===keys.length&&keys.every(k=>Object.prototype.hasOwnProperty.call(p,k)&&integer(p[k],signed.includes(k)?-1000000:k==='timeNumerator'?0:1,k==='sampleSteps'?64:1000000))
    &&Math.abs(p.angularNumerator)<=4*p.angularDenominator
    &&Math.abs(p.forceXNumerator)<=4*p.forceDenominator&&Math.abs(p.forceYNumerator)<=4*p.forceDenominator
    &&Math.abs(p.impulseXNumerator)<=4*p.impulseDenominator&&Math.abs(p.impulseYNumerator)<=4*p.impulseDenominator
    &&p.timeNumerator<=4*p.timeDenominator;
}
export function thermalOuResponseModel(input:ThermalOuResponseParameters) {
  if(!isThermalOuResponseParameters(input))throw new RangeError('Fifteen own integer fields: mu,kappa,theta and denominators positive1..1000000; omega and force/impulse components bounded by4 times their denominator; time0..4s; sampleSteps1..64.');
  const p=Object.freeze({...input}),q=(n:number,d:number)=>fraction(BigInt(n),BigInt(d));
  const mu=q(p.mobilityNumerator,p.coefficientDenominator),kappa=q(p.stiffnessNumerator,p.coefficientDenominator),theta=q(p.thermalEnergyNumerator,p.coefficientDenominator),omega=q(p.angularNumerator,p.angularDenominator);
  const gamma=mul(mu,kappa),diffusion=mul(mu,theta),variance=div(theta,kappa),T=q(p.timeNumerator,p.timeDenominator),dt=div(T,fraction(BigInt(p.sampleSteps)));
  const force=[q(p.forceXNumerator,p.forceDenominator),q(p.forceYNumerator,p.forceDenominator)],impulse=[q(p.impulseXNumerator,p.impulseDenominator),q(p.impulseYNumerator,p.impulseDenominator)];
  const denominator=add(mul(gamma,gamma),mul(omega,omega)),staticC=div(mul(mu,gamma),denominator),staticS=div(mul(mu,omega),denominator);
  const ctx=createTranscendentalContext(256),binary=(coefficient:bigint)=>({coefficient,exponent:0});
  const logarithm=(a:F):I=>{const r=ctx.logRatio(binary(a.n),binary(a.d));return{lower:fraction(r.lower,ctx.scale),upper:fraction(r.upper,ctx.scale)};};
  const logMu=logarithm(mu);
  const logStatus=(trig:I)=>trig.lower.n===Z&&trig.upper.n===Z?'structural-zero' as const:containsZero(trig)?'unresolved-trigonometric-factor' as const:'available' as const;
  function logAbsoluteResponse(trig:I,decayExponent:F) {
    if(trig.lower.n===Z&&trig.upper.n===Z||containsZero(trig))return null;
    const absolute=trig.upper.n<Z?scale(trig,fraction(-O)):trig;
    return observe(minus(plus(logMu,{lower:logarithm(absolute.lower).lower,upper:logarithm(absolute.upper).upper}),point(decayExponent)));
  }
  function diagnostic(sin:I,cos:I,diagonal:boolean) {
    const den=diagonal?cos:sin;
    if(den.lower.n===Z&&den.upper.n===Z)return {value:null,status:'undefined-zero-response' as const};
    if(containsZero(den))return {value:null,status:'unresolved-response-denominator' as const};
    const correction=scale(ratio(diagonal?sin:cos,den),div(omega,gamma));
    return {value:observe(scale(diagonal?plus(point(one),correction):minus(point(one),correction),theta)),status:'available' as const};
  }
  const history=[];
  for(let step=0;step<=p.sampleSteps;step++) {
    const t=mul(dt,fraction(BigInt(step))),angle=mul(omega,t),decayExponent=mul(gamma,t),rotation=directedRationalRotation(angle.n,angle.d);
    const sin:I={lower:fraction(rotation.sin.lower,rotation.scale),upper:fraction(rotation.sin.upper,rotation.scale)},cos:I={lower:fraction(rotation.cos.lower,rotation.scale),upper:fraction(rotation.cos.upper,rotation.scale)};
    const e=ctx.expNegativeRatio(binary(decayExponent.n),binary(decayExponent.d)),envelope:I={lower:fraction(e.lower,ctx.scale),upper:fraction(e.upper,ctx.scale)};
    const c=product(envelope,cos),s=product(envelope,sin),responseC=scale(c,mu),responseS=scale(s,mu);
    const derivativeC=scale(plus(scale(c,neg(gamma)),scale(s,neg(omega))),variance),derivativeS=scale(plus(scale(s,neg(gamma)),scale(c,omega)),variance);
    const currentC=scale(s,mul(omega,variance)),currentS=scale(c,neg(mul(omega,variance)));
    const naiveC=scale(derivativeC,neg(div(one,theta))),naiveS=scale(derivativeS,neg(div(one,theta)));
    const errorC=scale(s,div(omega,kappa)),errorS=scale(c,neg(div(omega,kappa)));
    const oneMinusC=minus(point(one),c);
    const chiC=scale(plus(scale(oneMinusC,gamma),scale(s,omega)),div(mu,denominator));
    const chiS=scale(minus(scale(oneMinusC,omega),scale(s,gamma)),div(mu,denominator));
    const naiveChiC=scale(oneMinusC,div(one,kappa)),naiveChiS=scale(s,neg(div(one,kappa)));
    const diagonal=diagnostic(sin,cos,true),offDiagonal=diagnostic(sin,cos,false);
    const logC=logAbsoluteResponse(cos,decayExponent),logS=logAbsoluteResponse(sin,decayExponent);
    history.push({step,time:output(t),angleRadians:output(angle),sine:observe(sin),cosine:observe(cos),relaxationEnvelope:observe(envelope),logRelaxationEnvelope:output(neg(decayExponent)),
      unperturbedCovXtX0:matrix(scale(c,variance),scale(s,variance)),covarianceLagDerivative:matrix(derivativeC,derivativeS),stationaryCurrentCorrelation:matrix(currentC,currentS),
      impulseResponse:matrix(responseC,responseS),naiveEquilibriumResponse:matrix(naiveC,naiveS),naiveResponseError:matrix(errorC,errorS),currentResponseCorrection:matrix(scale(errorC,fraction(-O)),scale(errorS,fraction(-O))),
      correctedResponse:matrix(responseC,responseS),scoreResponse:matrix(responseC,responseS),
      responseIdentityResidual:matrix(point(zero),point(zero)),
      stepSusceptibility:matrix(chiC,chiS),naiveEquilibriumStepSusceptibility:matrix(naiveChiC,naiveChiS),
      stepMean:vector(chiC,chiS,force),impulseMean:vector(responseC,responseS,impulse),naiveStepMean:vector(naiveChiC,naiveChiS,force),naiveImpulseMean:vector(naiveC,naiveS,impulse),
      logAbsoluteImpulseResponse:[[logC,logS],[logS,logC]],
      absoluteResponseLogStatus:[[logStatus(cos),logStatus(sin)],[logStatus(sin),logStatus(cos)]],
      effectiveThermalEnergy:[[diagonal.value,offDiagonal.value],[offDiagonal.value,diagonal.value]],
      effectiveThermalEnergyStatus:[[diagonal.status,offDiagonal.status],[offDiagonal.status,diagonal.status]],
      tZeroIsRightLimit:t.n===Z});
  }
  return {parameters:p,precisionBits:256,mobility:output(mu),stiffness:output(kappa),thermalEnergy:output(theta),relaxationRate:output(gamma),diffusion:output(diffusion),stationaryCoordinateVariance:output(variance),angularRate:output(omega),
    force:force.map(output),impulse:impulse.map(output),initialImpulseMean:impulse.map(v=>output(mul(mu,v))),staticSusceptibility:matrix(point(staticC),point(staticS)),staticMean:vector(point(staticC),point(staticS),force),
    stationaryCovarianceResidual:matrix(point(zero),point(zero)),history,current:history[history.length-1],
    classification:{stationaryInitialization:true,thermalEinsteinRelationSpecified:true,coordinatesAreEvenPositions:true,backgroundDetailedBalance:omega.n===Z,backgroundDriven:omega.n!==Z,forceResponseIsCausal:true,rightLimitAtZero:true,meanResponseIsExactlyLinearInThisFamily:true,sampledPaths:false,effectiveThermalEnergyIsThermometer:false},
    approximationContract:'Explicit thermal whole-plane Gaussian OU family: theta=kBT>0, mu,kappa>0, D=mu theta, gamma=mu kappa, A=−gamma I+omega R, R=[[0,−1],[1,0]]. Stationary even position initialization independent of future Brownian increments. Constant step force[N] and ideal impulse integral[Ns] are distinct inputs. Matrices use order E[Xt X0^T], positive lag derivatives and retarded response mu exp(At); time0 denotes its right limit. Exact rational coefficients, directed256bit exp/sin/cos/log and interval matrix expressions. Corrected/score response and zero identity residual use algebraic simplification, not subtraction of rounded projections or a claim about arbitrary measured data. A linear enclosure may be unresolved while a separate log envelope/absolute-response log remains available. Effective thermal energy is an observable/lag diagnostic; a zero response denominator is undefined, not infinity or a thermometer. Its common positive exponential cancels analytically before division, allowing a ratio even when a linear response display is unresolved. Means are exact for deterministic forcing within this affine model; no finite-perturbation claim for general nonlinear dynamics, no estimated temperatures, no sampled trajectories and no Euler approximation.'};
}
