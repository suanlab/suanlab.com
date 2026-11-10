/** Independent Q(sqrt(2),i) reference. Only rational primitives are shared.
 * No circuit, channel, density or scalar-wire producer is imported. */
import { f, zero, one, add, sub, mul, div, neg, eq, type F } from './learning-quantum-hilbert-core';
import { informationRequire as ensure, informationRecord as record, informationList as list } from './learning-quantum-information-response-checks';
export type ReferenceQuadratic=readonly[F,F];
export type ReferenceFieldComplex={re:ReferenceQuadratic;im:ReferenceQuadratic};
export const fieldRational=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
export const fieldPair=(a=zero,b=zero):ReferenceQuadratic=>[a,b];
const qa=(x:ReferenceQuadratic,y:ReferenceQuadratic):ReferenceQuadratic=>[add(x[0],y[0]),add(x[1],y[1])];
const qn=(x:ReferenceQuadratic):ReferenceQuadratic=>[neg(x[0]),neg(x[1])];
const qm=(x:ReferenceQuadratic,y:ReferenceQuadratic):ReferenceQuadratic=>[add(mul(x[0],y[0]),mul(f(BigInt(2)),mul(x[1],y[1]))),add(mul(x[0],y[1]),mul(x[1],y[0]))];
const qd=(x:ReferenceQuadratic,y:ReferenceQuadratic):ReferenceQuadratic=>{
 const denominator=sub(mul(y[0],y[0]),mul(f(BigInt(2)),mul(y[1],y[1])));
 if(eq(denominator,zero))throw new RangeError('Nonzero reference field denominator required');
 const numerator=qm(x,[y[0],neg(y[1])]);return[div(numerator[0],denominator),div(numerator[1],denominator)];
};
export const fieldComplex=(re=fieldPair(),im=fieldPair()):ReferenceFieldComplex=>({re,im});
export const fieldZero=fieldComplex(),fieldOne=fieldComplex(fieldPair(one));
export const fieldAdd=(x:ReferenceFieldComplex,y:ReferenceFieldComplex)=>fieldComplex(qa(x.re,y.re),qa(x.im,y.im));
export const fieldConjugate=(x:ReferenceFieldComplex)=>fieldComplex(x.re,qn(x.im));
export const fieldMultiply=(x:ReferenceFieldComplex,y:ReferenceFieldComplex)=>fieldComplex(qa(qm(x.re,y.re),qn(qm(x.im,y.im))),qa(qm(x.re,y.im),qm(x.im,y.re)));
export const fieldDivideReal=(x:ReferenceFieldComplex,y:ReferenceQuadratic)=>fieldComplex(qd(x.re,y),qd(x.im,y));
export const fieldEqual=(x:ReferenceFieldComplex,y:ReferenceFieldComplex)=>[0,1].every(i=>eq(x.re[i],y.re[i])&&eq(x.im[i],y.im[i]));
export const fieldNorm=(v:readonly ReferenceFieldComplex[]):ReferenceQuadratic=>v.reduce((s,z)=>qa(s,qa(qm(z.re,z.re),qm(z.im,z.im))),fieldPair());
export function referenceFieldVector(v:readonly {real:readonly string[];imaginary:readonly string[]}[]):ReferenceFieldComplex[]{
 const result=v.map(z=>fieldComplex([fieldRational(z.real[0]),fieldRational(z.real[1])],[fieldRational(z.imaginary[0]),fieldRational(z.imaginary[1])]));
 if(result.every(z=>fieldEqual(z,fieldZero)))throw new RangeError('Nonzero reference vector required');return result;
}
export const fieldDensity=(v:readonly ReferenceFieldComplex[])=>{const norm=fieldNorm(v);return v.map(a=>v.map(b=>fieldDivideReal(fieldMultiply(a,fieldConjugate(b)),norm)));};
export const fieldRationalWire=(x:F)=>x.d===BigInt(1)?String(x.n):`${x.n}/${x.d}`;
export const fieldWire=(z:ReferenceFieldComplex)=>({real:z.re.map(fieldRationalWire),imaginary:z.im.map(fieldRationalWire)});
export const fieldMatrixWire=(a:ReferenceFieldComplex[][])=>a.map(row=>row.map(fieldWire));
/** Exact canonical wire contract. Expected values come from the independent
 * reference, not the producer. Caller must own the untrusted packet first. */
export function checkInformationExactTree(actual:unknown,expected:unknown,path:string):number {
 if(Array.isArray(expected))return list(actual,expected.length,path).reduce<number>((n,x,i)=>n+checkInformationExactTree(x,expected[i],`${path}[${i}]`),0);
 if(expected&&typeof expected==='object'){
  const e=expected as Record<string,unknown>,a=record(actual,Object.keys(e),path);
  return Object.keys(e).reduce((n,k)=>n+checkInformationExactTree(a[k],e[k],`${path}.${k}`),0);
 }
 ensure(actual===expected,path,'Exact reference field or classification differs');
 return typeof expected==='string'&&/^-?\d+(?:\/\d+)?$/.test(expected)?1:0;
}
