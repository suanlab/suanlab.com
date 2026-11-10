/** Directed real2x2 flow and circular coordinates for the two-oscillator model.
 * Internal rational endpoints; public observations are serialized separately. */
import {fraction as f,zero,one,add,sub,mul,div,neg,equal,floor,ceil,observeRange,type F} from './learning-entropy-production-math';
import {directedPhaseRotation} from './learning-directed-phase-rotation';
export {f,zero,one,add,sub,mul,div,neg,equal,floor,ceil};export type {F};
export type R={lower:F;upper:F};
export const two=f(BigInt(2)),point=(x:F):R=>({lower:x,upper:x}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
export const plus=(a:R,b:R):R=>({lower:add(a.lower,b.lower),upper:add(a.upper,b.upper)});
export const negative=(a:R):R=>({lower:neg(a.upper),upper:neg(a.lower)}),minus=(a:R,b:R)=>plus(a,negative(b));
export const scale=(a:R,b:F):R=>b.n>=BigInt(0)?{lower:mul(a.lower,b),upper:mul(a.upper,b)}:{lower:mul(a.upper,b),upper:mul(a.lower,b)};
export const product=(a:R,b:R):R=>{const ends=[mul(a.lower,b.lower),mul(a.lower,b.upper),mul(a.upper,b.lower),mul(a.upper,b.upper)];return{lower:ends.reduce((x,y)=>le(x,y)?x:y),upper:ends.reduce((x,y)=>le(x,y)?y:x)};};
export const square=(a:R):R=>{const aa=mul(a.lower,a.lower),bb=mul(a.upper,a.upper);return{lower:a.lower.n<=BigInt(0)&&a.upper.n>=BigInt(0)?zero:le(aa,bb)?aa:bb,upper:le(aa,bb)?bb:aa};};
export function quotient(a:R,b:R):R{if(b.lower.n<=BigInt(0))throw new RangeError('Positive denominator enclosure required');return product(a,{lower:div(one,b.upper),upper:div(one,b.lower)});}
export const observation=(r:R)=>observeRange(r.lower,r.upper);
const S=BigInt(1)<<BigInt(256),W=BigInt(1)<<BigInt(288),Z=BigInt(0),O=BigInt(1);
function isqrt(n:bigint){if(n<BigInt(2))return n;let x=O<<BigInt(Math.ceil(n.toString(2).length/2));for(;;){const y=(x+n/x)/BigInt(2);if(y>=x)return x;x=y;}}
export function sqrt(a:R):R{if(a.lower.n<Z)throw new RangeError('Nonnegative square-root enclosure required');const bound=(x:F,upper:boolean)=>{const nr=isqrt(x.n),dr=isqrt(x.d);if(nr*nr===x.n&&dr*dr===x.d)return f(nr,dr);const target=x.n*S*S,r=isqrt(target/x.d);return f(r+(upper&&r*r*x.d!==target?O:Z),S);};return{lower:bound(a.lower,false),upper:bound(a.upper,true)};}
type I={lower:bigint;upper:bigint};
const ip=(n:bigint):I=>({lower:n,upper:n}),iadd=(a:I,b:I):I=>({lower:a.lower+b.lower,upper:a.upper+b.upper}),ineg=(a:I):I=>({lower:-a.upper,upper:-a.lower});
const imul=(a:I,b:I):I=>{const ends=[a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper];return{lower:floor(ends.reduce((a,b)=>a<b?a:b),W),upper:ceil(ends.reduce((a,b)=>a>b?a:b),W)};};
const idiv=(a:I,n:bigint):I=>({lower:floor(a.lower,n),upper:ceil(a.upper,n)}),ifraction=(a:F):I=>({lower:floor(a.n*W,a.d),upper:ceil(a.n*W,a.d)});
function atanSmall(x:F):R{
 if(x.n===Z)return point(zero);if(x.n<Z||x.n*BigInt(2)>x.d)throw new RangeError('atan series argument must be within0..1/2');
 const a=ifraction(x),aa=imul(a,a);let power=a,sum=ip(Z);const terms=160;
 for(let j=0;j<terms;j++){const term=idiv(power,BigInt(2*j+1));sum=iadd(sum,j%2?ineg(term):term);power=imul(power,aa);}
 const tail=ceil(W,(O<<BigInt(2*terms+1))*BigInt(2*terms+1));return{lower:f(floor(sum.lower-tail,BigInt(1)<<BigInt(32)),S),upper:f(ceil(sum.upper+tail,BigInt(1)<<BigInt(32)),S)};
}
export const piRange:Readonly<R>=Object.freeze(minus(scale(atanSmall(f(O,BigInt(5))),f(BigInt(16))),scale(atanSmall(f(O,BigInt(239))),f(BigInt(4)))));
function atanPoint(x:F):R{
 if(x.n<Z)return negative(atanPoint(neg(x)));if(x.n===Z)return point(zero);
 if(!le(x,one))return minus(scale(piRange,f(O,BigInt(2))),atanPoint(div(one,x)));
 if(x.n*BigInt(2)<=x.d)return atanSmall(x);
 return minus(scale(piRange,f(O,BigInt(4))),atanSmall(div(sub(one,x),add(one,x))));
}
export const atan=(r:R):R=>({lower:atanPoint(r.lower).lower,upper:atanPoint(r.upper).upper});
export function rotation(angle:R){const mid=div(add(angle.lower,angle.upper),two),radius=div(sub(angle.upper,angle.lower),two),r=directedPhaseRotation(mid.n,mid.d,256);const expand=(i:I):R=>{const lo=sub(f(i.lower,r.scale),radius),hi=add(f(i.upper,r.scale),radius);return{lower:le(lo,neg(one))?neg(one):lo,upper:le(one,hi)?one:hi};};return{cos:expand(r.cos),sin:expand(r.sin)};}
export type Matrix=R[][];
const exactMatrix=(m:F[][]):Matrix=>m.map(row=>row.map(point));
const identity=()=>exactMatrix([[one,zero],[zero,one]]);
const abs=(x:F)=>x.n<Z?neg(x):x;
/** exp(tA), A=[[K/2,-Delta/2],[Delta/2,-K/2]].
 * Critical A^2=0 is evaluated exactly; otherwise signed Taylor scaling/squaring.
 */
export function pairFlowMatrix(K:F,delta:F,t:F):Matrix{
 if(t.n===Z)return identity();const h=f(O,BigInt(2)),A=[[mul(K,h),neg(mul(delta,h))],[mul(delta,h),neg(mul(K,h))]];
 if(equal(mul(K,K),mul(delta,delta)))return A.map((row,i)=>row.map((q,j)=>point(add(i===j?one:zero,mul(t,q)))));
 const norm=mul(mul(add(K,abs(delta)),h),t);let squarings=0,powerOfTwo=O;while(norm.n*BigInt(8)>norm.d*powerOfTwo){squarings++;powerOfTwo*=BigInt(2);}
 type IM=I[][];const multiply=(a:IM,b:IM):IM=>a.map(row=>b[0].map((_,j)=>row.reduce((sum,x,k)=>iadd(sum,imul(x,b[k][j])),ip(Z))));
 const B=A.map(row=>row.map(q=>ifraction(div(mul(q,t),f(powerOfTwo)))));let term:IM=[[ip(W),ip(Z)],[ip(Z),ip(W)]],result:IM=term;
 for(let j=1;j<=64;j++){term=multiply(term,B).map(row=>row.map(q=>idiv(q,BigInt(j))));result=result.map((row,i)=>row.map((q,k)=>iadd(q,term[i][k])));}
 let factorial=O;for(let j=2;j<=65;j++)factorial*=BigInt(j);const tail=ceil(BigInt(2)*W,(O<<BigInt(3*65))*factorial);result=result.map(row=>row.map(q=>({lower:q.lower-tail,upper:q.upper+tail})));
 for(let j=0;j<squarings;j++)result=multiply(result,result);
 return result.map(row=>row.map(q=>({lower:f(q.lower,W),upper:f(q.upper,W)})));
}
export function matrixVector(M:Matrix,v:F[]):R[]{return M.map(row=>row.reduce((sum,q,i)=>plus(sum,scale(q,v[i])),point(zero)));}
