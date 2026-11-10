/** Exact complex/polynomial algebra and directed512-bit normalization. */
import {markovFraction as f,exactMarkovObservation as exact,type MarkovFraction as F}from'./learning-finite-markov-exact';
import {partitionObservationFromRange}from'./learning-partition-exact-core';
export {f,exact};export type{F};
export const zero=f(BigInt(0)),one=f(BigInt(1)),two=f(BigInt(2)),add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),eq=(a:F,b:F)=>a.n*b.d===b.n*a.d,le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
export const sum=(xs:F[])=>xs.reduce(add,zero),q=(x:F)=>exact(x.n,x.d);
export type C={re:F;im:F};export const c=(re:F,im=zero):C=>({re,im}),ca=(a:C,b:C)=>c(add(a.re,b.re),add(a.im,b.im)),cn=(a:C)=>c(neg(a.re),neg(a.im)),cs=(a:C,b:C)=>ca(a,cn(b)),cc=(a:C)=>c(a.re,neg(a.im)),cm=(a:C,b:C)=>c(sub(mul(a.re,b.re),mul(a.im,b.im)),add(mul(a.re,b.im),mul(a.im,b.re))),scaleC=(a:C,b:F)=>c(mul(a.re,b),mul(a.im,b)),abs2=(a:C)=>add(mul(a.re,a.re),mul(a.im,a.im)),norm2=(xs:C[])=>sum(xs.map(abs2)),dot=(a:C[],b:C[])=>a.reduce((s,x,i)=>ca(s,cm(cc(x),b[i])),c(zero)),cq=(a:C)=>({real:q(a.re),imaginary:q(a.im)});
export type R={lower:F;upper:F};const S=BigInt(1)<<BigInt(512);
function isqrt(n:bigint){if(n<BigInt(2))return n;let x=BigInt(1)<<BigInt(Math.ceil(n.toString(2).length/2));for(;;){const y=(x+n/x)/BigInt(2);if(y>=x)return x;x=y;}}
export function sqrtRange(x:F):R{if(x.n<BigInt(0))throw new RangeError('Nonnegative square root required');const a=isqrt(x.n),b=isqrt(x.d);if(a*a===x.n&&b*b===x.d)return{lower:f(a,b),upper:f(a,b)};const scaled=x.n*S*S,v=isqrt(scaled/x.d);return{lower:f(v,S),upper:f(v+(v*v*x.d===scaled?BigInt(0):BigInt(1)),S)};}
export function observe(r:R){if(eq(r.lower,r.upper))return q(r.lower);const b=(x:F)=>({n:{coefficient:x.n,exponent:0},d:{coefficient:x.d,exponent:0}});return partitionObservationFromRange({lower:b(r.lower),upper:b(r.upper)});}
export function overRoot(x:F,normSquared:F):R{if(normSquared.n<=BigInt(0))throw new RangeError('Positive normalization norm required');if(x.n===BigInt(0))return{lower:zero,upper:zero};const root=sqrtRange(normSquared);if(root.lower.n===BigInt(0))throw new RangeError('Normalization root unresolved at512bits');return x.n>BigInt(0)?{lower:div(x,root.upper),upper:div(x,root.lower)}:{lower:div(x,root.lower),upper:div(x,root.upper)};}
export const normalized=(z:C,normSquared:F)=>({real:observe(overRoot(z.re,normSquared)),imaginary:observe(overRoot(z.im,normSquared))});
export type P=F[];
export const polyAdd=(a:P,b:P):P=>Array.from({length:Math.max(a.length,b.length)},(_,i)=>add(a[i]??zero,b[i]??zero)),polyScale=(a:P,s:F)=>a.map(x=>mul(x,s)),polySub=(a:P,b:P)=>polyAdd(a,polyScale(b,neg(one)));
export function polyMul(a:P,b:P):P{const r=Array.from({length:a.length+b.length-1},()=>zero);a.forEach((x,i)=>b.forEach((y,j)=>{r[i+j]=add(r[i+j],mul(x,y));}));return r;}
export const polyValue=(p:P,x:F)=>p.reduceRight((s,a)=>add(mul(s,x),a),zero);
export function integral(p:P,a:F,b:F):F{const primitive=[zero,...p.map((v,i)=>div(v,f(BigInt(i+1))))];return sub(polyValue(primitive,b),polyValue(primitive,a));}
export const squaredMagnitudePolynomial=(re:P,im:P)=>polyAdd(polyMul(re,re),polyMul(im,im));
