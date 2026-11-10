/** Exact complex Gram algebra for the nonorthogonal Ritz engine. */
import {variationalDeterminant} from './learning-quantum-variational-math';
import {c,zero,one,f,add,mul,div,neg,cs,cc,cm,scaleC,norm2,type C,type F} from './learning-quantum-hilbert-core';
import {adjoint,product,identity,trace,type Matrix} from './learning-observable-matrix';
export function ritzGram(B:Matrix):Matrix{return product(adjoint(B),B);}
export function ritzInverseGram(S:Matrix):Matrix {
 const determinant=variationalDeterminant(S);
 if(determinant.im.n!==BigInt(0)||determinant.re.n<=BigInt(0))throw new RangeError('Independent trial columns with a positive Gram determinant required');
 return S.map((row,i)=>row.map((_,j)=>scaleC(variationalDeterminant(S.filter((_,r)=>r!==j).map(row=>row.filter((_,k)=>k!==i))),div((i+j)%2?neg(one):one,determinant.re))));
}
export function ritzCharacteristic(C:Matrix):F[]{
 const n=C.length,traces:F[]=[];let power=identity(n);
 for(let j=1;j<=n;j++){power=product(power,C);const t=trace(power);if(t.im.n!==BigInt(0))throw new RangeError('Real characteristic traces required');traces.push(t.re);}
 const e=[one];for(let j=1;j<=n;j++){let v=zero;for(let i=1;i<=j;i++)v=add(v,mul(e[j-i],i%2?traces[i-1]:neg(traces[i-1])));e.push(div(v,f(BigInt(j))));}
 return e.map((v,i)=>i%2?neg(v):v);
}
export const ritzComplexDivide=(a:C,b:C)=>scaleC(cm(a,cc(b)),div(one,norm2([b])));
export const ritzVectorSubtract=(a:C[],b:C[])=>a.map((v,i)=>cs(v,b[i]));
export const ritzZeroMatrix=(n:number,m=n):Matrix=>Array.from({length:n},()=>Array.from({length:m},()=>c(zero)));
