/** Certified amplitudes and branch-preserving real-time wave observations.
 * Exact matrices remain in the finite core. Large/small amplitude is carried
 * with a shared binary scale and, for Abel damping, a separate exponential.
 */
import {buildRealTimeChainExact,buildRealTimePacketExact} from './learning-real-time-phase-exact';
import {f,two,add,mul,div,neg,abs2,type F,type C} from './learning-quantum-hilbert-core';
import {createRealTimeIntervalContext,realTimeFloor,realTimeCeil,type RealTimeInterval as I} from './learning-real-time-interval';
const ctx=createRealTimeIntervalContext(8192),Z=BigInt(0),O=BigInt(1);
type CI={real:I;imaginary:I};
const complex=(real:I,imaginary=ctx.zero):CI=>({real,imaginary});
const product=(a:CI,b:CI):CI=>({real:ctx.sub(ctx.multiply(a.real,b.real),ctx.multiply(a.imaginary,b.imaginary)),imaginary:ctx.add(ctx.multiply(a.real,b.imaginary),ctx.multiply(a.imaginary,b.real))});
const scale=(a:CI,b:I):CI=>({real:ctx.multiply(a.real,b),imaginary:ctx.multiply(a.imaginary,b)});
const phase=(a:I):CI=>{const v=ctx.sinCos(a);return complex(v.cos,v.sin);};
const absolute=(x:F)=>x.n<Z?neg(x):x;
const half=()=>ctx.root(ctx.divideInteger(ctx.one,2),2);
function octant(k:number):CI {
  const h=half(),negative=ctx.neg(h),plus=ctx.one,minus=ctx.neg(plus),z=ctx.zero;
  return [complex(plus,z),complex(h,negative),complex(z,minus),complex(negative,negative),complex(minus,z),complex(negative,h),complex(z,plus),complex(h,h)][((k%8)+8)%8];
}
/** Principal sqrt of a positive-real-part complex number, normalized and
 * conjugated. Using Im/(2*|z|*cosHalf) avoids subtractive half-angle loss.
 */
function inverseRootPhase(z:C):CI {
  if(z.re.n<=Z)throw new RangeError('Positive-real Gaussian pivot required');
  const magnitude=ctx.root(ctx.from(abs2(z)),2),cosine=ctx.root(ctx.divide(ctx.add(magnitude,ctx.from(z.re)),ctx.scale(magnitude,2)),2);
  const sine=ctx.divide(ctx.from(z.im),ctx.scale(ctx.multiply(magnitude,cosine),2));
  return complex(cosine,ctx.neg(sine));
}
const observeComplex=(z:CI)=>({real:ctx.observe(z.real),imaginary:ctx.observe(z.imaginary)});
function commonScale(values:I[]) {
  let max=Z;for(const x of values)for(const a of [x.lower,x.upper]){const v=a<Z?-a:a;if(v>max)max=v;}
  const exponent=max===Z?0:max.toString(2).length-ctx.workBits;
  const normalize=(x:I):I=>exponent>=0?{lower:realTimeFloor(x.lower,O<<BigInt(exponent)),upper:realTimeCeil(x.upper,O<<BigInt(exponent))}:{lower:x.lower*(O<<BigInt(-exponent)),upper:x.upper*(O<<BigInt(-exponent))};
  return{exponent,normalize};
}
function scaledComplex(z:CI) {
  const s=commonScale([z.real,z.imaginary]);return{scalePowerOfTwo:s.exponent,...observeComplex(complex(s.normalize(z.real),s.normalize(z.imaginary)))};
}
function scaledPositive(x:I) {
  const s=commonScale([x]),range=ctx.observe(s.normalize(x));
  return{scalePowerOfTwo:s.exponent,lower:range.lower,upper:range.upper};
}
export function observeRealTimeChainWave(r:ReturnType<typeof buildRealTimeChainExact>) {
  let unregulated;
  if(r.inertia.zero) {
    unregulated={kind:'parity-delta' as const,parity:r.causticIndex!%2?-1:1,maslovIndex:r.causticIndex!,unitPhase:observeComplex(octant(2*r.causticIndex!)),pointwiseValue:null};
  } else {
    const squaredTimesPi=div(r.s,mul(mul(two,r.hbar),absolute(r.theta[r.d])));
    const amplitude=ctx.root(ctx.divide(ctx.from(squaredTimesPi),ctx.pi()),2);
    const unit=product(phase(ctx.from(div(r.stationaryAction!,r.hbar))),octant(1+2*r.inertia.negative));
    const logMagnitude=ctx.divideInteger(ctx.sub(ctx.log(ctx.from(squaredTimesPi)),ctx.log(ctx.pi())),2);
    unregulated={kind:'finite-quadratic-kernel' as const,amplitude:scaledPositive(amplitude),unitPhase:observeComplex(unit),value:scaledComplex(scale(unit,amplitude)),logMagnitude:ctx.observe(logMagnitude)};
  }
  let regulated=null;
  if(r.regulator) {
    const determinantMagnitude=ctx.root(ctx.from(abs2(r.regulator.determinant)),2);
    const power=f(r.s.n**BigInt(r.n),r.s.d**BigInt(r.n));
    const amplitudeSquared=ctx.divide(ctx.from(div(power,mul(two,r.hbar))),ctx.multiply(ctx.pi(),determinantMagnitude));
    const amplitude=ctx.root(amplitudeSquared,2);
    let unit=octant(r.n);for(const pivot of r.regulator.pivots)unit=product(unit,inverseRootPhase(pivot));
    unit=product(unit,phase(ctx.from(r.regulator.exponent.im)));
    if(r.regulator.exponent.re.n>Z)throw Error('Abel Gaussian exponent cannot amplify its centered integral');
    const decay=ctx.from(neg(r.regulator.exponent.re)),scaled=scale(unit,amplitude),raw=scale(scaled,ctx.expNegative(decay));
    const logMagnitude=ctx.sub(ctx.divideInteger(ctx.log(amplitudeSquared),2),decay);
    regulated={kind:'abel-regulated-kernel' as const,scaledAmplitude:scaledPositive(amplitude),unitPhase:observeComplex(unit),scaledValue:scaledComplex(scaled),value:scaledComplex(raw),negativeExponential:ctx.observe(decay),
      logMagnitude:ctx.observe(logMagnitude),rawValueRequiresScaledRepresentation:raw.real.lower<=Z&&raw.real.upper>=Z&&raw.imaginary.lower<=Z&&raw.imaginary.upper>=Z,
      branch:'product-of-positive-real-pivot-square-roots' as const};
  }
  return{unregulated,regulated,workingBits:ctx.workBits,observationBits:512};
}
export function observeRealTimePacketWave(r:ReturnType<typeof buildRealTimePacketExact>) {
  const variance=ctx.from(r.finalCovariance[0][0]),normalization=ctx.divide(ctx.one,ctx.root(ctx.multiply(ctx.scale(ctx.pi(),2),variance),4));
  const M=r.map.total,B=M[0][1];let metaplectic:CI;
  if(B.n===Z) {
    if(r.causticIndex===null)throw Error('A zero B map requires its exact caustic index');
    metaplectic=octant(2*r.causticIndex);
  } else {
    const cg={re:div(r.hbar,mul(two,r.variance)),im:neg(add(div(M[0][0],B),r.chirp))};
    metaplectic=product(octant(1+2*r.negativeModes),inverseRootPhase(cg));
  }
  const centerPhase=product(metaplectic,phase(ctx.from(div(r.centralAction,r.hbar))));
  const centerValue=scale(centerPhase,normalization),standardDeviation=ctx.root(variance,2);
  const hbar=ctx.from(r.hbar),position=ctx.from(r.center[0]),momentum=ctx.from(r.center[1]),chirp=ctx.from(r.finalWidth.re);
  const values=[];
  for(let index=0;index<=64;index++) {
    const offset=f(BigInt(index-32),BigInt(8)),offsetI=ctx.from(offset),dx=ctx.multiply(offsetI,standardDeviation);
    const envelope=ctx.expNegative(ctx.divideInteger(ctx.square(offsetI),4));
    const localAction=ctx.add(ctx.multiply(momentum,dx),ctx.divideInteger(ctx.multiply(chirp,ctx.square(dx)),2));
    const unit=product(centerPhase,phase(ctx.divide(localAction,hbar)));
    const wave=scale(unit,ctx.multiply(normalization,envelope)),density=ctx.multiply(ctx.square(normalization),ctx.square(envelope));
    values.push({standardDeviations:index/8-4,position:ctx.observe(ctx.add(position,dx)),unitPhase:observeComplex(unit),value:scaledComplex(wave),density:scaledPositive(density)});
  }
  return{normalization:scaledPositive(normalization),logNormalization:ctx.observe(ctx.neg(ctx.divideInteger(ctx.log(ctx.multiply(ctx.scale(ctx.pi(),2),variance)),4))),
    metaplecticPhase:observeComplex(metaplectic),centerPhase:observeComplex(centerPhase),centerValue:scaledComplex(centerValue),
    standardDeviation:scaledPositive(standardDeviation),samples:values,positionVariance:'final-symmetrized-covariance-qq' as const,
    sampleCoordinates:'mean-plus-standard-deviations' as const,workingBits:ctx.workBits,observationBits:512};
}
