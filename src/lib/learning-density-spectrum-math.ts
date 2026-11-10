/** Certified spectral scalars; exact endpoints are classified before logarithms. */
import {createQuantumEvolutionContext,type EvolutionInterval as I} from './learning-quantum-evolution-interval';
import {createTranscendentalContext} from './learning-interval-transcendentals';
import {f,one,zero,two,sub,eq,le,type F} from './learning-quantum-hilbert-core';
export function createDensitySpectrumMath(){
 const ctx=createQuantumEvolutionContext(),logs=createTranscendentalContext(768),b=(n:bigint)=>({coefficient:n,exponent:0}),logRational=(x:F)=>logs.logRatio(b(x.n),b(x.d)),ln2Fixed=logRational(two),ln2=ctx.range({lower:f(ln2Fixed.lower,logs.scale),upper:f(ln2Fixed.upper,logs.scale)});
 const logPositive=(x:I):I=>{if(x.lower<=BigInt(0))throw new RangeError('Positive spectral interval required; exact zero must be classified separately');const range=ctx.rationalRange(x),lo=logRational(range.lower),hi=logRational(range.upper);return ctx.range({lower:f(lo.lower,logs.scale),upper:f(hi.upper,logs.scale)});};
 const calculate=(radiusSquared:F)=>{
  if(!le(zero,radiusSquared)||!le(radiusSquared,one))throw new RangeError('Exact squared radius in[0,1] required');
  const pure=eq(radiusSquared,one),maximallyMixed=eq(radiusSquared,zero),radius=ctx.sqrt(radiusSquared),high=ctx.divide(ctx.add(ctx.one,radius),ctx.from(two)),low=pure?ctx.zero:ctx.divide(ctx.from(sub(one,radiusSquared)),ctx.multiply(ctx.from(two),ctx.add(ctx.one,radius))),lowLog=pure?null:logPositive(low),highLog=logPositive(high),terms=[lowLog===null?ctx.zero:ctx.neg(ctx.multiply(low,lowLog)),ctx.neg(ctx.multiply(high,highLog))],nats=ctx.add(terms[0],terms[1]),bits=pure?ctx.zero:maximallyMixed?ctx.one:ctx.divide(nats,ln2);
  return{radius,low,high,logs:[lowLog,highLog],terms,nats,bits,pure,maximallyMixed};
 };
 const cache=new Map<string,ReturnType<typeof calculate>>(),binary=(r2:F)=>{const key=r2.n+'/'+r2.d;let v=cache.get(key);if(!v){v=calculate(r2);cache.set(key,v);}return v;};
 return{ctx,ln2,binary,logWorkBits:800,logReportedBits:768};
}
