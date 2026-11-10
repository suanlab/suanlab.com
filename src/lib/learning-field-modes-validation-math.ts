/** Independent3072-bit grid and2048-bit pi; no production mode/trig engine. */
import {f,int,zero,one,two,add,sub,mul,div,neg,eq,le,type F,type Range} from './learning-tensor-wire-math';
import {wireRange as point,wireAdd as plus,wireSub as minus,wireNeg as negative,wireMul as times,wireDiv as quotient,wireSquare as squared,wireSqrt as root} from './learning-many-body-schmidt-interval-wire';
import {directedPi} from './learning-directed-pi';
export {f,int,zero,one,two,add,sub,mul,div,neg,eq,le,point,plus,minus,negative,times,quotient,squared,root};export type{F,Range};
export const parse=(s:string):F=>{if(!/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(s)||s.length>4096)throw Error('Canonical bounded fraction required');const[n,d='1']=s.split('/'),x=f(BigInt(n),BigInt(d));if(text(x)!==s)throw Error('Reduced fraction required');return x;};
export const text=(v:F)=>v.d===BigInt(1)?String(v.n):`${v.n}/${v.d}`;
export const inputFraction=(s:string)=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
export const total=(xs:Range[])=>xs.reduce(plus,point(zero)),scale=(x:Range,c:F)=>times(x,point(c)),rationalTotal=(xs:F[])=>xs.reduce(add,zero);
export const pi=()=>{const x=directedPi(2048),S=BigInt(1)<<BigInt(2048);return point(f(x.lower,S),f(x.upper,S));};
export function trigContext(){
 const cache=new Map<string,{sin:Range;cos:Range}>(),P=pi();
 function reducedSineCosine(t:F){
  if(eq(t,zero))return{sin:point(zero),cos:point(one)};
  if(eq(t,div(one,two)))return{sin:point(one),cos:point(zero)};
  const x=scale(P,t),x2=squared(x);let sin=point(zero),cos=point(zero),st=x,ct=point(one);const terms=224;
  for(let j=0;j<terms;j++){sin=j%2?minus(sin,st):plus(sin,st);cos=j%2?minus(cos,ct):plus(cos,ct);st=scale(times(st,x2),f(BigInt(1),BigInt((2*j+2)*(2*j+3))));ct=scale(times(ct,x2),f(BigInt(1),BigInt((2*j+1)*(2*j+2))));}
  // |x|<2. Taylor remainders are <=2^(2N+1)/(2N+1)! and2^(2N)/(2N)!.
  let factorial=BigInt(1);for(let j=2;j<=2*terms;j++)factorial*=BigInt(j);
  const cb=f(BigInt(1)<<BigInt(2*terms),factorial),sb=f(BigInt(1)<<BigInt(2*terms+1),factorial*BigInt(2*terms+1));
  return{sin:plus(sin,point(neg(sb),sb)),cos:plus(cos,point(neg(cb),cb))};
 }
 return(turn:F)=>{const period=BigInt(2)*turn.d,r=f((turn.n%period+period)%period,turn.d),key=text(r);const old=cache.get(key);if(old)return old;
  const half=div(one,two),threeHalf=div(int(3),two);let t:F,ss=1,cs=1;if(le(r,half))t=r;else if(le(r,one)){t=sub(one,r);cs=-1;}else if(le(r,threeHalf)){t=sub(r,one);ss=-1;cs=-1;}else{t=sub(two,r);ss=-1;}const value=reducedSineCosine(t),answer={sin:ss<0?negative(value.sin):value.sin,cos:cs<0?negative(value.cos):value.cos};cache.set(key,answer);return answer;};
}
