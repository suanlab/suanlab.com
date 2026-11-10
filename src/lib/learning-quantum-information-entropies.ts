/** Directed entropy/logarithm arithmetic in bits, for finite rational qubits. */
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { f, zero, one, two, add, sub, mul, div, le, eq, sqrtRange, sum, type F, type R } from './learning-quantum-hilbert-core';
export const point = (x: F): R => ({lower:x,upper:x});
export const plus = (a:R,b:R):R=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
export const minus = (a:R,b:R):R=>({lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)});
const min=(a:F,b:F)=>le(a,b)?a:b,max=(a:F,b:F)=>le(a,b)?b:a;
export function times(a:R,b:R):R {
  const values=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];
  return {lower:values.reduce(min),upper:values.reduce(max)};
}
export const scale=(a:R,x:F)=>times(a,point(x));
export function divide(a:R,b:R):R {
  if(le(b.lower,zero))throw new RangeError('Positive resolved denominator required');
  return times(a,{lower:div(one,b.upper),upper:div(one,b.lower)});
}
export function intersect(a:R,lower=zero,upper=one):R {
  const result={lower:max(lower,a.lower),upper:min(upper,a.upper)};
  if(!le(result.lower,result.upper))throw new RangeError('Analytic range and enclosure disagree');
  return result;
}
const context=createTranscendentalContext(256);
const binary=(x:bigint)=>({coefficient:x,exponent:0});
function ln(x:F):R {
  const r=context.logRatio(binary(x.n),binary(x.d));
  return {lower:f(r.lower,context.scale),upper:f(r.upper,context.scale)};
}
const lnTwo=ln(two);
export function log2Range(x:R):R {
  if(le(x.lower,zero))throw new RangeError('Positive resolved logarithm argument required');
  if(eq(x.lower,one)&&eq(x.upper,one))return point(zero);
  return divide({lower:ln(x.lower).lower,upper:ln(x.upper).upper},lnTwo);
}
function hPoint(x:F):R {
  if(eq(x,zero)||eq(x,one))return point(zero);
  if(eq(mul(two,x),one))return point(one);
  const other=sub(one,x);
  return scale(plus(scale(log2Range(point(x)),x),scale(log2Range(point(other)),other)),f(BigInt(-1)));
}
export function binaryEntropyRange(x:R):R {
  if(!le(zero,x.lower)||!le(x.lower,x.upper)||!le(x.upper,one))throw new RangeError('Probability interval required');
  const a=hPoint(x.lower),b=hPoint(x.upper),half=div(one,two);
  return intersect({lower:min(a.lower,b.lower),upper:le(x.lower,half)&&le(half,x.upper)?one:max(a.upper,b.upper)});
}
export const dot=(a:readonly F[],b:readonly F[])=>sum(a.map((x,i)=>mul(x,b[i])));
export function qubitEntropyRange(r:readonly F[]):R {
  const radius=sqrtRange(dot(r,r));
  return binaryEntropyRange(intersect(scale(minus(point(one),radius),div(one,two)),zero,div(one,two)));
}
