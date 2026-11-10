/** Exact small Hermitian invariants; all principal minors, not only leading ones. */
import {c,ca,cc,cm,scaleC,f,zero,one,neg,sum,type C,type F} from './learning-quantum-hilbert-core';
import type {Matrix} from './learning-observable-matrix';

export function mixedDeterminant(a:Matrix):C {
 return a.length===0?c(one):a[0].reduce((s,v,j)=>ca(s,scaleC(cm(v,mixedDeterminant(a.slice(1).map(row=>row.filter((_,k)=>k!==j)))),j%2?neg(one):one)),c(zero));
}
export function mixedHermitianInvariants(a:Matrix) {
 const d=a.length,principalMinors=Array.from({length:2**d-1},(_,mask)=>{
  const indices=Array.from({length:d},(_,i)=>i).filter(i=>((mask+1)&2**i)!==0);
  const determinant=mixedDeterminant(indices.map(i=>indices.map(j=>a[i][j])));
  if(determinant.im.n!==BigInt(0))throw new RangeError('Real Hermitian principal minors required');
  return {indices,determinant:determinant.re};
 });
 const e=[one,...Array.from({length:d},(_,i)=>sum(principalMinors.filter(m=>m.indices.length===i+1).map(m=>m.determinant)))];
 return {principalMinors,positiveSemidefinite:principalMinors.every(m=>m.determinant.n>=BigInt(0)),
  characteristicPolynomial:e.map((v,i)=>i%2?neg(v):v),
  determinant:principalMinors.at(-1)!.determinant,
  // This is rank only when the matrix is PSD; do not apply it to NPT spectra.
  positivePrincipalRank:Math.max(0,...principalMinors.filter(m=>m.determinant.n>BigInt(0)).map(m=>m.indices.length))};
}
export function mixedInputMatrix(p:Readonly<Record<string,number>>):Matrix {
 const integer=(n:number):F=>f(BigInt(n));
 return Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>{
  const u=Math.min(i,j),v=Math.max(i,j),z=c(integer(p[`hReal${u}${v}`]),i===j?zero:integer(p[`hImag${u}${v}`]));
  return i>j?cc(z):z;
 }));
}
