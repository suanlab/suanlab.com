/** Small exact complex matrices. Used only by the quantum observable engines. */
import {c,ca,cs,cc,cm,scaleC,dot,norm2,zero,one,sub,mul,div,sum,eq,q,cq,normalized,type C,type F} from './learning-quantum-hilbert-core';
export type Matrix=C[][];
export const identity=(n:number):Matrix=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>c(i===j?one:zero)));
export const matrixAdd=(a:Matrix,b:Matrix):Matrix=>a.map((row,i)=>row.map((v,j)=>ca(v,b[i][j])));
export const matrixSub=(a:Matrix,b:Matrix):Matrix=>a.map((row,i)=>row.map((v,j)=>cs(v,b[i][j])));
export const matrixScale=(a:Matrix,s:F):Matrix=>a.map(row=>row.map(v=>scaleC(v,s)));
export const adjoint=(a:Matrix):Matrix=>a[0].map((_,j)=>a.map(row=>cc(row[j])));
export const apply=(a:Matrix,z:C[]):C[]=>a.map(row=>row.reduce((s,v,j)=>ca(s,cm(v,z[j])),c(zero)));
export const product=(a:Matrix,b:Matrix):Matrix=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>ca(s,cm(v,b[k][j])),c(zero))));
export const outer=(z:C[]):Matrix=>z.map(a=>z.map(b=>cm(a,cc(b))));
export const projector=(z:C[]):Matrix=>matrixScale(outer(z),div(one,norm2(z)));
export const trace=(a:Matrix):C=>a.reduce((s,row,i)=>ca(s,row[i]),c(zero));
export const matrixObservation=(a:Matrix)=>a.map(row=>row.map(cq));
export const matrixSquaredNorm=(a:Matrix):F=>sum(a.map(norm2));
export const isZeroMatrix=(a:Matrix)=>a.every(row=>row.every(v=>v.re.n===BigInt(0)&&v.im.n===BigInt(0)));
export type SpectralPart={eigenvalue:F;rank:number;projector:Matrix};
/** Merge equal eigenvalues before interpreting an ideal spectral measurement. */
export function spectralParts(p:Matrix,positive:F,negative:F):SpectralPart[]{
 const n=p.length;
 if(eq(positive,negative))return[{eigenvalue:positive,rank:n,projector:identity(n)}];
 const parts=[{eigenvalue:positive,rank:1,projector:p},{eigenvalue:negative,rank:n-1,projector:matrixSub(identity(n),p)}];
 return parts.sort((a,b)=>{const d=sub(a.eigenvalue,b.eigenvalue);return d.n<BigInt(0)?-1:1;});
}
export function spectralMeasurement(z:C[],parts:SpectralPart[]){
 const n=z.length,N=norm2(z),A=parts.reduce((s,p)=>matrixAdd(s,matrixScale(p.projector,p.eigenvalue)),matrixScale(identity(n),zero));
 const branches=parts.map(p=>{const projected=apply(p.projector,z),mass=norm2(projected),probability=div(mass,N);return{...p,projected,mass,probability};});
 const density=branches.reduce((s,b)=>matrixAdd(s,matrixScale(outer(b.projected),div(one,N))),matrixScale(identity(n),zero));
 const mean=sum(branches.map(b=>mul(b.eigenvalue,b.probability))),second=sum(branches.map(b=>mul(mul(b.eigenvalue,b.eigenvalue),b.probability))),variance=sub(second,mul(mean,mean));
 const directMean=scaleC(dot(z,apply(A,z)),div(one,N)),directSecond=div(norm2(apply(A,z)),N);
 const summed=parts.reduce((s,p)=>matrixAdd(s,p.projector),matrixScale(identity(n),zero));
 return{A,branches,density,mean,second,variance,observation:{matrix:matrixObservation(A),outcomes:branches.map(b=>({eigenvalue:q(b.eigenvalue),rank:b.rank,projector:matrixObservation(b.projector),projectedNumerators:b.projected.map(cq),probability:q(b.probability),conditionalAmplitudes:b.mass.n===BigInt(0)?null:b.projected.map(v=>normalized(v,b.mass)),expectationContribution:q(mul(b.eigenvalue,b.probability))})),mean:q(mean),secondMoment:q(second),variance:q(variance),nonselectiveDensity:matrixObservation(density),nonselectivePurity:q(trace(product(density,density)).re),residuals:{hermiticity:matrixObservation(matrixSub(A,adjoint(A))),projectorIdempotence:parts.map(p=>matrixObservation(matrixSub(product(p.projector,p.projector),p.projector))),projectorHermiticity:parts.map(p=>matrixObservation(matrixSub(p.projector,adjoint(p.projector)))),projectorOrthogonality:parts.map((p,i)=>parts.map((r,j)=>i===j?null:matrixObservation(product(p.projector,r.projector)))),projectorCompleteness:matrixObservation(matrixSub(summed,identity(n))),probabilityMass:q(sub(sum(branches.map(b=>b.probability)),one)),directMean:cq(cs(directMean,c(mean))),directSecondMoment:q(sub(directSecond,second)),nonselectiveTrace:cq(cs(trace(density),c(one)))}}};
}
