/** Directed768-bit work arithmetic, outward512-bit observations.
 * sin/cos use |x/2^k|<=1/8, alternating Taylor bounds and k doublings;
 * no floating trigonometry, decimal angle reduction or norm correction.
 */
import {f,zero,one,add as fa,sub as fs,div as fd,eq,le,sqrtRange,observe,type F,type R,type C} from './learning-quantum-hilbert-core';
export type EvolutionInterval={lower:bigint;upper:bigint};
export type EvolutionComplex={real:EvolutionInterval;imaginary:EvolutionInterval};
export type EvolutionMatrix=EvolutionComplex[][];
const Z=BigInt(0),O=BigInt(1),T=BigInt(2),workBits=768,outputBits=512,S=O<<BigInt(workBits),shrink=O<<BigInt(workBits-outputBits),limit=O<<BigInt(64);
const floor=(n:bigint,d:bigint):bigint=>d<Z?floor(-n,-d):n>=Z?n/d:-((-n+d-O)/d),ceil=(n:bigint,d:bigint)=>-floor(-n,d),min=(xs:bigint[])=>xs.reduce((a,b)=>a<b?a:b),max=(xs:bigint[])=>xs.reduce((a,b)=>a>b?a:b);
function valid(x:F){if(!x||typeof x.n!=='bigint'||typeof x.d!=='bigint'||x.d<=Z||x.n.toString(2).length>65536||x.d.toString(2).length>65536)throw new RangeError('Bounded rational with positive denominator required');}
export function createQuantumEvolutionContext(){
 const point=(x:bigint):EvolutionInterval=>({lower:x,upper:x}),from=(x:F):EvolutionInterval=>{valid(x);return{lower:floor(x.n*S,x.d),upper:ceil(x.n*S,x.d)};},neg=(x:EvolutionInterval):EvolutionInterval=>({lower:-x.upper,upper:-x.lower}),add=(a:EvolutionInterval,b:EvolutionInterval):EvolutionInterval=>({lower:a.lower+b.lower,upper:a.upper+b.upper}),sub=(a:EvolutionInterval,b:EvolutionInterval)=>add(a,neg(b));
 const multiply=(a:EvolutionInterval,b:EvolutionInterval):EvolutionInterval=>{const xs=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];return{lower:floor(min(xs),S),upper:ceil(max(xs),S)};};
 const divide=(a:EvolutionInterval,b:EvolutionInterval):EvolutionInterval=>{if(b.lower<=Z&&b.upper>=Z)throw new RangeError('Division interval contains zero');const ns=[a.lower,a.lower,a.upper,a.upper],ds=[b.lower,b.upper,b.lower,b.upper];return{lower:min(ns.map((n,i)=>floor(n*S,ds[i]))),upper:max(ns.map((n,i)=>ceil(n*S,ds[i])))};};
 const square=(a:EvolutionInterval):EvolutionInterval=>{const xs=[a.lower*a.lower,a.upper*a.upper];return{lower:a.lower<=Z&&a.upper>=Z?Z:floor(min(xs),S),upper:ceil(max(xs),S)};};
 const integerDivide=(a:EvolutionInterval,d:bigint):EvolutionInterval=>({lower:floor(a.lower,d),upper:ceil(a.upper,d)}),twice=(a:EvolutionInterval):EvolutionInterval=>({lower:T*a.lower,upper:T*a.upper});
 const intersect=(a:EvolutionInterval,lower:bigint,upper:bigint):EvolutionInterval=>{const result={lower:a.lower<lower?lower:a.lower,upper:a.upper>upper?upper:a.upper};if(result.lower>result.upper)throw new RangeError('Certified interval is disjoint from a known mathematical range');return result;};
 const cache=new Map<string,{sin:EvolutionInterval;cos:EvolutionInterval}>(),freeze=(v:{sin:EvolutionInterval;cos:EvolutionInterval})=>Object.freeze({sin:Object.freeze({...v.sin}),cos:Object.freeze({...v.cos})});
 function sinCosPoint(x:F){
  valid(x);if(x.n>Z?x.n>limit*x.d:-x.n>limit*x.d)throw new RangeError('Phase magnitude exceeds2^64 radians');
  const key=x.n+'/'+x.d,found=cache.get(key);if(found)return found;
  if(x.n===Z)return freeze({sin:point(Z),cos:point(S)});
  const negative=x.n<Z,n=negative?-x.n:x.n;let k=0,d=x.d;while(n*BigInt(8)>d){d*=T;k++;}
  const angle=from(f(n,d)),x2=square(angle),terms=Math.ceil((workBits+32)/6);let st=angle,ct=point(S),sin=angle,cos=point(S);
  for(let j=1;j<=terms;j++){st=integerDivide(multiply(st,x2),BigInt(2*j)*BigInt(2*j+1));ct=integerDivide(multiply(ct,x2),BigInt(2*j-1)*BigInt(2*j));sin=j%2?sub(sin,st):add(sin,st);cos=j%2?sub(cos,ct):add(cos,ct);}
  // Both first omitted terms are <=2^(-6*terms-3) for |angle|<=1/8.
  const tail=ceil(S,O<<BigInt(6*terms+3));sin={lower:sin.lower-tail,upper:sin.upper+tail};cos={lower:cos.lower-tail,upper:cos.upper+tail};
  for(let j=0;j<k;j++){const nextSin=twice(multiply(sin,cos)),nextCos=sub(square(cos),square(sin));sin=intersect(nextSin,-S,S);cos=intersect(nextCos,-S,S);}
  const result=freeze({sin:negative?neg(sin):intersect(sin,-S,S),cos:intersect(cos,-S,S)});cache.set(key,result);return result;
 }
 function sinCos(x:R){
  valid(x.lower);valid(x.upper);if(!le(x.lower,x.upper)||!le(f(-limit),x.lower)||!le(x.upper,f(limit)))throw new RangeError('Ordered phase interval within±2^64 required');
  if(eq(x.lower,x.upper))return sinCosPoint(x.lower);
  // |sin'(x)|,|cos'(x)|<=1: widen the exact midpoint's result by the
  // input radius. Do not propagate argument uncertainty through k doublings.
  const radius=from(fd(fs(x.upper,x.lower),f(T))).upper;if(radius>=T*S)return{sin:{lower:-S,upper:S},cos:{lower:-S,upper:S}};
  const middle=fd(fa(x.lower,x.upper),f(T)),v=sinCosPoint(middle),widen=(a:EvolutionInterval)=>intersect({lower:a.lower-radius,upper:a.upper+radius},-S,S);return{sin:widen(v.sin),cos:widen(v.cos)};
 }
 const range=(x:R):EvolutionInterval=>{valid(x.lower);valid(x.upper);if(!le(x.lower,x.upper))throw new RangeError('Ordered endpoints required');return{lower:from(x.lower).lower,upper:from(x.upper).upper};};
 const observation=(x:EvolutionInterval)=>{if(x.lower>x.upper)throw new RangeError('Ordered endpoints required');return observe({lower:f(floor(x.lower,shrink),O<<BigInt(outputBits)),upper:f(ceil(x.upper,shrink),O<<BigInt(outputBits))});};
 const rationalRange=(x:EvolutionInterval):R=>({lower:f(x.lower,S),upper:f(x.upper,S)}),complex=(z:C):EvolutionComplex=>({real:from(z.re),imaginary:from(z.im)}),complexAdd=(a:EvolutionComplex,b:EvolutionComplex):EvolutionComplex=>({real:add(a.real,b.real),imaginary:add(a.imaginary,b.imaginary)}),complexNeg=(z:EvolutionComplex):EvolutionComplex=>({real:neg(z.real),imaginary:neg(z.imaginary)}),complexSubtract=(a:EvolutionComplex,b:EvolutionComplex)=>complexAdd(a,complexNeg(b)),conjugate=(z:EvolutionComplex):EvolutionComplex=>({real:z.real,imaginary:neg(z.imaginary)}),complexMultiply=(a:EvolutionComplex,b:EvolutionComplex):EvolutionComplex=>({real:sub(multiply(a.real,b.real),multiply(a.imaginary,b.imaginary)),imaginary:add(multiply(a.real,b.imaginary),multiply(a.imaginary,b.real))}),complexScale=(z:EvolutionComplex,a:EvolutionInterval):EvolutionComplex=>({real:multiply(z.real,a),imaginary:multiply(z.imaginary,a)}),absSquared=(z:EvolutionComplex)=>add(square(z.real),square(z.imaginary));
 const cz=()=>complex({re:zero,im:zero}),identity=(n:number):EvolutionMatrix=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>complex({re:i===j?one:zero,im:zero}))),apply=(a:EvolutionMatrix,z:EvolutionComplex[])=>a.map(row=>row.reduce((s,v,j)=>complexAdd(s,complexMultiply(v,z[j])),cz())),product=(a:EvolutionMatrix,b:EvolutionMatrix):EvolutionMatrix=>a.map(row=>b[0].map((_,j)=>row.reduce((s,v,k)=>complexAdd(s,complexMultiply(v,b[k][j])),cz()))),adjoint=(a:EvolutionMatrix):EvolutionMatrix=>a[0].map((_,j)=>a.map(row=>conjugate(row[j]))),matrixSubtract=(a:EvolutionMatrix,b:EvolutionMatrix)=>a.map((row,i)=>row.map((v,j)=>complexSubtract(v,b[i][j]))),normSquared=(z:EvolutionComplex[])=>z.reduce((s,v)=>add(s,absSquared(v)),point(Z)),dot=(a:EvolutionComplex[],b:EvolutionComplex[])=>a.reduce((s,v,i)=>complexAdd(s,complexMultiply(conjugate(v),b[i])),cz()),observeComplex=(z:EvolutionComplex)=>({real:observation(z.real),imaginary:observation(z.imaginary)}),observeMatrix=(a:EvolutionMatrix)=>a.map(row=>row.map(observeComplex));
 return{workBits,precisionBits:outputBits,from,range,rationalRange,zero:point(Z),one:point(S),neg,add,sub,multiply,divide,square,intersect,probability:(x:EvolutionInterval)=>observation(intersect(x,Z,S)),observation,sinCos,sqrt:(x:F)=>range(sqrtRange(x)),complex,complexAdd,complexSubtract,complexMultiply,complexScale,conjugate,absSquared,identity,apply,product,adjoint,matrixSubtract,normSquared,dot,observeComplex,observeMatrix};
}
