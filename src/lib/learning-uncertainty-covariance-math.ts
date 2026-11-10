/** Directed small covariance spectra; classifications use exact rational invariants. */
import {createQuantumEvolutionContext} from './learning-quantum-evolution-interval';
import {f,c,zero,two,add,sub,mul,div,neg,eq,le,abs2,sqrtRange,q,type F,type C} from './learning-quantum-hilbert-core';
export function createUncertaintyCovarianceMath(){
 const ctx=createQuantumEvolutionContext(),square=(x:F)=>mul(x,x),absolute=(x:F)=>x.n<BigInt(0)?neg(x):x;
 const root=(x:F)=>{const range=sqrtRange(x),known=eq(range.lower,range.upper)?range.lower:null;return{interval:ctx.range(range),observation:known===null?ctx.observation(ctx.range(range)):q(known),known};};
 // Normalize a component before taking the root. No division by a tiny rounded gap.
 const overRoot=(x:F,normSquared:F)=>{if(normSquared.n<=BigInt(0))throw new RangeError('Positive exact squared norm required');if(x.n===BigInt(0))return ctx.zero;const v=root(div(square(x),normSquared)).interval;return x.n<BigInt(0)?ctx.neg(v):v;};
 function spectrum(a:F,b:F,z:C){
  const trace=add(a,b),determinant=sub(mul(a,b),abs2(z)),difference=sub(a,b),gapSquared=add(square(difference),mul(f(BigInt(4)),abs2(z)));
  if(!le(zero,a)||!le(zero,b)||!le(zero,determinant))throw new RangeError('Exact positive semidefinite covariance required');
  const original=[[c(a),z],[c(z.re,neg(z.im)),c(b)]],I=ctx.identity(2),source=original.map(row=>row.map(ctx.complex)),merged=gapSquared.n===BigInt(0),rank=trace.n===BigInt(0)?0:determinant.n===BigInt(0)?1:2;
  let parts:{value:ReturnType<typeof ctx.from>;known:F|null;rank:number;projector:ReturnType<typeof ctx.identity>;zeroExact:boolean}[];
  if(merged){const value=div(trace,two);parts=[{value:ctx.from(value),known:value,rank:2,projector:I,zeroExact:value.n===BigInt(0)}];}
  else{
   const gap=root(gapSquared),highKnown=gap.known===null?null:div(add(trace,gap.known),two),lowKnown=gap.known===null?null:div(sub(trace,gap.known),two);
   const high=determinant.n===BigInt(0)?ctx.from(trace):ctx.divide(ctx.add(ctx.from(trace),gap.interval),ctx.from(two)),low=determinant.n===BigInt(0)?ctx.zero:ctx.divide(ctx.from(determinant),high);
   const d=overRoot(difference,gapSquared),re=overRoot(mul(two,z.re),gapSquared),im=overRoot(mul(two,z.im),gapSquared),D=[[{real:d,imaginary:ctx.zero},{real:re,imaginary:im}],[{real:re,imaginary:ctx.neg(im)},{real:ctx.neg(d),imaginary:ctx.zero}]];
   const projector=(sign:number)=>D.map((row,i)=>row.map((v,j)=>({real:ctx.divide(ctx.add(i===j?ctx.one:ctx.zero,sign===1?v.real:ctx.neg(v.real)),ctx.from(two)),imaginary:ctx.divide(sign===1?v.imaginary:ctx.neg(v.imaginary),ctx.from(two))})));
   parts=[{value:low,known:determinant.n===BigInt(0)?zero:lowKnown,rank:1,projector:projector(-1),zeroExact:determinant.n===BigInt(0)},{value:high,known:determinant.n===BigInt(0)?trace:highKnown,rank:1,projector:projector(1),zeroExact:false}];
  }
  const sum=(matrices:ReturnType<typeof ctx.identity>[])=>matrices.reduce((a,b)=>a.map((row,i)=>row.map((v,j)=>ctx.complexAdd(v,b[i][j])))),reconstructed=sum(parts.map(p=>p.projector.map(row=>row.map(v=>ctx.complexScale(v,p.value)))));
  return{trace:q(trace),determinant:q(determinant),gapSquared:q(gapSquared),positiveRank:rank,eigenspacesMerged:merged,parts:parts.map(p=>({eigenvalue:p.known===null?ctx.observation(p.value):q(p.known),rank:p.rank,projector:ctx.observeMatrix(p.projector),zeroEigenvalueExact:p.zeroExact})),directedResiduals:{reconstruction:ctx.observeMatrix(ctx.matrixSubtract(reconstructed,source)),projectorCompleteness:ctx.observeMatrix(ctx.matrixSubtract(sum(parts.map(p=>p.projector)),I)),idempotence:parts.map(p=>ctx.observeMatrix(ctx.matrixSubtract(ctx.product(p.projector,p.projector),p.projector))),hermiticity:parts.map(p=>ctx.observeMatrix(ctx.matrixSubtract(p.projector,ctx.adjoint(p.projector)))),eigenEquation:parts.map(p=>ctx.observeMatrix(ctx.matrixSubtract(ctx.product(source,p.projector),p.projector.map(row=>row.map(v=>ctx.complexScale(v,p.value))))))}};
 }
 return{ctx,root,signedOverRoot:overRoot,absolute,spectrum};
}
