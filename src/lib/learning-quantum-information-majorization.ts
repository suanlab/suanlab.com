/** Single-copy bipartite pure-state LOCC certificates on Schmidt probabilities. */
import { f, zero, one, add, sub, mul, div, le, eq, sum, q, observe, type F } from './learning-quantum-hilbert-core';
import { point, plus, scale, log2Range, intersect } from './learning-quantum-information-entropies';
function spectrum(input:readonly F[]){
 if(!Array.isArray(input)||input.length<1||input.length>16)throw new RangeError('Use one to16 Schmidt probabilities');
 for(const x of Array.from(input))if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.d<=BigInt(0)||x.n.toString().replace('-','').length>32||x.d.toString().length>32||!le(zero,x))throw new RangeError('Nonnegative rational probabilities with at most32 digits required');
 if(!eq(sum(Array.from(input)),one))throw new RangeError('Schmidt probabilities must sum exactly to one');
 return input.map((value,index)=>({value,index})).sort((a,b)=>eq(a.value,b.value)?a.index-b.index:le(a.value,b.value)?1:-1);
}
function direction(source:F[],target:F[]){
 let a=zero,b=zero;
 const prefixes=source.map((x,i)=>{a=add(a,x);b=add(b,target[i]);return {k:i+1,source:q(a),target:q(b),targetMinusSource:q(sub(b,a)),satisfied:le(a,b)};});
 let optimum=one;
 const tails=source.map((_,i)=>{
  const numerator=sum(source.slice(i)),denominator=sum(target.slice(i));
  if(eq(denominator,zero))return {first:i+1,source:q(numerator),target:q(denominator),applicable:false,ratio:null};
  const ratio=div(numerator,denominator);if(le(ratio,optimum))optimum=ratio;
  return {first:i+1,source:q(numerator),target:q(denominator),applicable:true,ratio:q(ratio)};
 });
 return {deterministic:prefixes.every(p=>p.satisfied),firstViolation:prefixes.find(p=>!p.satisfied)?.k??null,prefixes,tails,optimalStochasticSuccess:q(optimum)};
}
function entropy(values:F[]){
 const terms=values.filter(x=>x.n>BigInt(0)).map(x=>scale(log2Range(point(x)),mul(f(BigInt(-1)),x)));
 const bound=log2Range(point(f(BigInt(values.length)))).upper;
 return observe(intersect(terms.reduce(plus,point(zero)),zero,bound));
}
export function pureStateMajorization(sourceInput:readonly F[],targetInput:readonly F[]){
 const sourceSorted=spectrum(sourceInput),targetSorted=spectrum(targetInput),dimension=Math.max(sourceSorted.length,targetSorted.length);
 const pad=(xs:ReturnType<typeof spectrum>)=>Array.from({length:dimension},(_,i)=>xs[i]?.value??zero);
 const source=pad(sourceSorted),target=pad(targetSorted),forward=direction(source,target),reverse=direction(target,source);
 return {dimension,source:source.map(q),target:target.map(q),sourceSortingIndices:sourceSorted.map(x=>x.index),targetSortingIndices:targetSorted.map(x=>x.index),
  sourceSchmidtRank:source.filter(x=>x.n>BigInt(0)).length,targetSchmidtRank:target.filter(x=>x.n>BigInt(0)).length,
  sourceEntropy:entropy(source),targetEntropy:entropy(target),forward,reverse,
  relation:forward.deterministic&&reverse.deterministic?'equivalent':forward.deterministic?'forward-only':reverse.deterministic?'reverse-only':'incomparable',
  contract:'Exact descending Schmidt probability majorization and Vidal tail ratios for one bipartite pure-state copy under unrestricted LOCC. Zero target tails are skipped. Entropies use directed256-bit logarithms and do not decide the partial order. This certifies existence/optimal probability by the applicable theorems, not an implemented general LOCC protocol. No mixed-state, catalytic, approximate, asymptotic or channel-capacity claim; no Worker/public input reader/UI.'};
}
