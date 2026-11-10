/** Directed512-bit stable real2x2 exponential with near-coalescence series.
 * No eigenvector diagonalization, positive large exponentials, or ODE stepping. */
import {markovFraction as f,exactMarkovObservation,type MarkovFraction as F}from'./learning-finite-markov-exact';
import {partitionObservationFromRange}from'./learning-partition-exact-core';
import {createTranscendentalContext}from'./learning-interval-transcendentals';
import {directedPhaseRotation}from'./learning-directed-phase-rotation';
export {f};export type{F};export const zero=f(BigInt(0)),one=f(BigInt(1)),two=f(BigInt(2)),add=(a:F,b:F)=>f(a.n*b.d+b.n*a.d,a.d*b.d),neg=(a:F)=>f(-a.n,a.d),sub=(a:F,b:F)=>add(a,neg(b)),mul=(a:F,b:F)=>f(a.n*b.n,a.d*b.d),div=(a:F,b:F)=>f(a.n*b.d,a.d*b.n),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d,abs=(a:F)=>a.n<BigInt(0)?neg(a):a;
export type R={lower:F;upper:F};export const point=(x:F):R=>({lower:x,upper:x});
const S=BigInt(1)<<BigInt(512),context=createTranscendentalContext(512),floor=(n:bigint,d:bigint)=>n>=BigInt(0)?n/d:-((-n+d-BigInt(1))/d),ceil=(n:bigint,d:bigint)=>-floor(-n,d);
export const rounded=(r:R):R=>({lower:f(floor(r.lower.n*S,r.lower.d),S),upper:f(ceil(r.upper.n*S,r.upper.d),S)});
export const plus=(a:R,b:R)=>rounded({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)}),negative=(a:R)=>({lower:neg(a.upper),upper:neg(a.lower)}),minus=(a:R,b:R)=>plus(a,negative(b));
export const scale=(a:R,b:F)=>rounded(b.n>=BigInt(0)?{lower:mul(a.lower,b),upper:mul(a.upper,b)}:{lower:mul(a.upper,b),upper:mul(a.lower,b)});
export function product(a:R,b:R):R{const v=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];return rounded({lower:v.reduce((x,y)=>le(x,y)?x:y),upper:v.reduce((x,y)=>le(x,y)?y:x)});}
export function quotient(a:R,b:R):R{if(b.lower.n<=BigInt(0))throw new RangeError('Strictly positive denominator enclosure required');return product(a,{lower:div(one,b.upper),upper:div(one,b.lower)});}
export function expNegative(r:R):R{
 if(r.upper.n>BigInt(0))throw new RangeError('Nonpositive exponent required');const endpoint=(x:F)=>{if(x.n===BigInt(0))return point(one);const q=context.expNegativeRatio({coefficient:-x.n,exponent:0},{coefficient:x.d,exponent:0});return{lower:f(q.lower,S),upper:f(q.upper,S)};};return{lower:endpoint(r.lower).lower,upper:endpoint(r.upper).upper};
}
function isqrt(n:bigint){if(n<BigInt(2))return n;let x=BigInt(1)<<BigInt(Math.ceil(n.toString(2).length/2));for(;;){const y=(x+n/x)/BigInt(2);if(y>=x)return x;x=y;}}
export function sqrt(x:F):R{if(x.n<BigInt(0))throw new RangeError('Nonnegative square root');const a=isqrt(x.n),b=isqrt(x.d);if(a*a===x.n&&b*b===x.d)return point(f(a,b));const n=x.n*S*S,v=isqrt(n/x.d);return{lower:f(v,S),upper:f(v+(v*v*x.d===n?BigInt(0):BigInt(1)),S)};}
export function observe(r:R){if(r.lower.n*r.upper.d===r.upper.n*r.lower.d)return exactMarkovObservation(r.lower.n,r.lower.d);const binary=(x:bigint)=>({coefficient:x,exponent:0});return partitionObservationFromRange({lower:{n:binary(r.lower.n),d:binary(r.lower.d)},upper:{n:binary(r.upper.n),d:binary(r.upper.d)}});}
export const containsZero=(r:R)=>r.lower.n<=BigInt(0)&&r.upper.n>=BigInt(0);
export type Matrix=R[][];
export function linearMemoryFlow(a:F,b:F,c:F,d:F,epsilon:F,t:F){
 const fast=div(d,epsilon),off=div(c,epsilon),tau=div(add(a,fast),two),difference=div(sub(fast,a),two),eta=add(mul(difference,difference),mul(b,off)),det=div(sub(mul(a,d),mul(b,c)),epsilon),B=[[difference,b],[off,neg(difference)]];
 if(t.n===BigInt(0))return{matrix:[[point(one),point(zero)],[point(zero),point(one)]],diagonal:point(one),coefficient:point(zero),method:'exact-zero-time' as const,tau,eta,det};
 const u=mul(eta,mul(t,t));let diagonal:R,coefficient:R,method:'coalescence-series'|'real-decaying-modes'|'damped-rotation';
 if(le(abs(u),f(BigInt(1),BigInt(16)))){
  // cosh(sqrt(u)) and sinh(sqrt(u))/sqrt(u) are entire in signed u.
  let term0=point(one),term1=point(one),sum0=point(one),sum1=point(one);
  for(let k=1;k<=64;k++){term0=scale(term0,div(u,f(BigInt((2*k-1)*(2*k)))));term1=scale(term1,div(u,f(BigInt((2*k)*(2*k+1)))));sum0=plus(sum0,term0);sum1=plus(sum1,term1);}
  let factorial=BigInt(1);for(let k=2;k<=130;k++)factorial*=BigInt(k);const tail=f(BigInt(2),(BigInt(16)**BigInt(65))*factorial),expand=(r:R)=>({lower:sub(r.lower,tail),upper:add(r.upper,tail)}),decay=expNegative(point(neg(mul(tau,t))));
  diagonal=product(decay,expand(sum0));coefficient=scale(product(decay,expand(sum1)),t);method='coalescence-series';
 }else if(eta.n>BigInt(0)){
  const root=sqrt(eta),denominator=plus(point(tau),root),slow=negative(quotient(point(det),denominator)),fastRate=negative(denominator),es=expNegative(scale(slow,t)),ef=expNegative(scale(fastRate,t));
  diagonal=scale(plus(es,ef),f(BigInt(1),BigInt(2)));coefficient=quotient(minus(es,ef),scale(root,two));method='real-decaying-modes';
 }else{
  const root=sqrt(neg(eta)),angle=scale(root,t),mid=div(add(angle.lower,angle.upper),two),radius=div(sub(angle.upper,angle.lower),two),rotation=directedPhaseRotation(mid.n,mid.d,512),expand=(v:{lower:bigint;upper:bigint})=>rounded({lower:sub(f(v.lower,rotation.scale),radius),upper:add(f(v.upper,rotation.scale),radius)}),decay=expNegative(point(neg(mul(tau,t))));
  diagonal=product(decay,expand(rotation.cos));coefficient=product(decay,quotient(expand(rotation.sin),root));method='damped-rotation';
 }
 return{diagonal,coefficient,matrix:B.map((row,i)=>row.map((value,j)=>plus(i===j?diagonal:point(zero),scale(coefficient,value)))),method,tau,eta,det};
}
export const matrixVector=(M:Matrix,v:F[])=>M.map(row=>row.reduce((s,r,i)=>plus(s,scale(r,v[i])),point(zero)));

export function logPositive(x:F):R{if(x.n<=BigInt(0))throw new RangeError('Positive logarithm input required');const r=context.logRatio({coefficient:x.n,exponent:0},{coefficient:x.d,exponent:0});return{lower:f(r.lower,S),upper:f(r.upper,S)};}
