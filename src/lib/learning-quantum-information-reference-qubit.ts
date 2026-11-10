/** Independent2x2 matrix/root/entropy reference. No producer model or root/log evaluator. */
import{f,zero,one,two,add,sub,mul,div,neg,eq,le,type F,type R}from'./learning-quantum-hilbert-core';
import{independentInformationEntropy}from'./learning-quantum-information-reference-log';
export type ReferenceComplex={re:F;im:F};export type ReferenceMatrix=ReferenceComplex[][];
const c=(re:F,im=zero):ReferenceComplex=>({re,im});
const cm=(a:ReferenceComplex,b:ReferenceComplex)=>c(sub(mul(a.re,b.re),mul(a.im,b.im)),add(mul(a.re,b.im),mul(a.im,b.re)));
export const referencePoint=(x:F):R=>({lower:x,upper:x});
export const referenceAdd=(a:R,b:R):R=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
export const referenceSubtract=(a:R,b:R):R=>({lower:sub(a.lower,b.upper),upper:sub(a.upper,b.lower)});
export const referenceScale=(a:R,x:F):R=>x.n<BigInt(0)?{lower:mul(a.upper,x),upper:mul(a.lower,x)}:{lower:mul(a.lower,x),upper:mul(a.upper,x)};
export function referenceClip(a:R,lower=zero,upper=one):R{
 const result={lower:le(a.lower,lower)?lower:a.lower,upper:le(upper,a.upper)?upper:a.upper};
 if(!le(result.lower,result.upper))throw new RangeError('Independent interval misses analytic range');return result;
}
function integerRoot(n:bigint):bigint{
 if(n<BigInt(2))return n;
 let low=BigInt(0),high=BigInt(1)<<BigInt(Math.ceil(n.toString(2).length/2));
 while(high-low>BigInt(1)){const mid=(low+high)/BigInt(2);if(mid*mid<=n)low=mid;else high=mid;}return low;
}
export function independentUnitRoot(x:F):R{
 if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.d<=BigInt(0))throw new RangeError('Positive rational denominator required');
 x=f(x.n,x.d);
 if(!le(zero,x)||!le(x,one))throw new RangeError('Unit-interval rational root required');
 const a=integerRoot(x.n),b=integerRoot(x.d);
 if(a*a===x.n&&b*b===x.d)return referencePoint(f(a,b));
 const scale=BigInt(1)<<BigInt(768),target=x.n*scale*scale;
 let low=BigInt(0),high=scale+BigInt(1);
 while(high-low>BigInt(1)){const mid=(low+high)/BigInt(2);if(mid*mid*x.d<=target)low=mid;else high=mid;}
 return{lower:f(low,scale),upper:f(low+BigInt(1),scale)};
}
export function referenceDensity(r:readonly F[]):ReferenceMatrix{
 if(r.length!==3)throw new RangeError('Three Bloch components required');
 const half=(x:F)=>div(x,two),rho=[[c(half(add(one,r[2]))),c(half(r[0]),neg(half(r[1])))],[c(half(r[0]),half(r[1])),c(half(sub(one,r[2])))]];
 if(!le(zero,rho[0][0].re)||!le(zero,rho[1][1].re)||!le(zero,referenceDeterminant(rho)))throw new RangeError('Nonphysical reference qubit');return rho;
}
export const referenceDeterminant=(rho:ReferenceMatrix)=>sub(mul(rho[0][0].re,rho[1][1].re),add(mul(rho[0][1].re,rho[0][1].re),mul(rho[0][1].im,rho[0][1].im)));
export const referenceTrace=(rho:ReferenceMatrix)=>add(rho[0][0].re,rho[1][1].re);
export const referenceTraceProduct=(a:ReferenceMatrix,b:ReferenceMatrix)=>a.flatMap((row,i)=>row.map((z,j)=>cm(z,b[j][i]).re)).reduce(add,zero);
export const referenceMatrixCombination=(a:ReferenceMatrix,x:F,b:ReferenceMatrix,y:F)=>a.map((row,i)=>row.map((z,j)=>c(add(mul(x,z.re),mul(y,b[i][j].re)),add(mul(x,z.im),mul(y,b[i][j].im)))));
export function referenceCommuting(a:ReferenceMatrix,b:ReferenceMatrix):boolean{
 const entry=(x:ReferenceMatrix,y:ReferenceMatrix,i:number,j:number)=>x[i].map((z,k)=>cm(z,y[k][j])).reduce((s,z)=>c(add(s.re,z.re),add(s.im,z.im)),c(zero));
 return[0,1].every(i=>[0,1].every(j=>{const x=entry(a,b,i,j),y=entry(b,a,i,j);return eq(x.re,y.re)&&eq(x.im,y.im);}));
}
export function referenceBinaryEntropy(probability:R):R{
 const p=referenceClip(probability),left=independentInformationEntropy([p.lower,sub(one,p.lower)]),right=independentInformationEntropy([p.upper,sub(one,p.upper)]),half=div(one,two);
 return referenceClip({lower:le(left.lower,right.lower)?left.lower:right.lower,upper:le(p.lower,half)&&le(half,p.upper)?one:le(left.upper,right.upper)?right.upper:left.upper});
}
export function referenceQubitEntropy(rho:ReferenceMatrix):R{
 const trace=referenceTrace(rho);if(!eq(trace,one))throw new RangeError('Unit-trace reference qubit required');
 const gap=independentUnitRoot(sub(mul(trace,trace),mul(f(BigInt(4)),referenceDeterminant(rho))));
 return referenceBinaryEntropy(referenceClip(referenceScale(referenceSubtract(referencePoint(one),gap),div(one,two)),zero,div(one,two)));
}
export function referenceHelstromProbabilities(rho0:ReferenceMatrix,rho1:ReferenceMatrix,prior:F){
 const delta=referenceMatrixCombination(rho0,prior,rho1,neg(sub(one,prior))),trace=referenceTrace(delta),det=referenceDeterminant(delta);
 if(le(zero,det))return{constant:true,probabilities:[referencePoint(le(zero,trace)?one:zero),referencePoint(le(zero,trace)?one:zero)]};
 const gap=independentUnitRoot(sub(mul(trace,trace),mul(f(BigInt(4)),det)));
 if(gap.lower.n<=BigInt(0))throw new RangeError('Reference spectral gap unresolved');
 const probability=(rho:ReferenceMatrix)=>{
  const numerator=sub(referenceTraceProduct(delta,rho),div(trace,two));
  const quotient=numerator.n<BigInt(0)?{lower:div(numerator,gap.lower),upper:div(numerator,gap.upper)}:{lower:div(numerator,gap.upper),upper:div(numerator,gap.lower)};
  return referenceClip(referenceAdd(referencePoint(div(one,two)),quotient));
 };
 return{constant:false,probabilities:[probability(rho0),probability(rho1)]};
}
