import {tensorRecord} from './learning-tensor-input';
import {isMasterObservation} from './learning-master-equation-protocol';
import {binaryRatioObservation} from './learning-binary-products';
import {f,one,two,zero,add,sub,mul,div,neg,le,lo,hi,type F} from './learning-tensor-wire-math';
import type {PartitionObservation as Q} from './learning-partition-exact-core';
export type Check=(x:unknown)=>boolean;
export const shape=(x:unknown,s:Record<string,Check>):x is Record<string,unknown>=>tensorRecord(x)&&Reflect.ownKeys(x).length===Object.keys(s).length&&Object.entries(s).every(([k,c])=>Object.hasOwn(x,k)&&c(x[k]));
export const literal=(v:unknown):Check=>x=>x===v,choice=(...v:unknown[]):Check=>x=>v.includes(x),bool:Check=x=>typeof x==='boolean',integer=(x:unknown,min=0,max=Number.MAX_SAFE_INTEGER):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
export const array=(n:number,c:Check):Check=>x=>Array.isArray(x)&&Object.getPrototypeOf(x)===Array.prototype&&x.length===n&&Reflect.ownKeys(x).length===n+1&&Array.from({length:n},(_,i)=>Object.hasOwn(x,i)&&c(x[i])).every(Boolean);
export const list=(min:number,max:number,c:Check):Check=>x=>Array.isArray(x)&&x.length>=min&&x.length<=max&&array(x.length,c)(x),nullable=(c:Check):Check=>x=>x===null||c(x),fields=(names:string,c:Check)=>Object.fromEntries(names.split(' ').map(k=>[k,c]));
export function createTensorWireChecks(){
 const cache=new WeakMap<object,boolean>(),term=(positive=false):Check=>x=>shape(x,{coefficient:v=>typeof v==='string'&&v.length<=4096&&(positive?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(v)&&v!=='-0',exponent:literal(0)}),fraction:Check=x=>shape(x,{numerator:term(),denominator:term(true)});
 const q:Check=x=>{
  if(!tensorRecord(x))return false;if(cache.has(x))return cache.get(x)!;
  let ok=shape(x,{value:v=>v===null||typeof v==='number'&&Number.isFinite(v),sign:choice(-1,0,1,null),logMagnitude:v=>v===null||typeof v==='number'&&Number.isFinite(v),logResolution:choice('finite','not-applicable','unresolved'),resolution:choice('structural-zero','exact-rational','interval-resolved','unresolved-at-precision-ceiling'),bounds:v=>shape(v,{lower:fraction,upper:fraction})})&&isMasterObservation(x);
  if(ok){const v=x as unknown as Q,a=lo(v),b=hi(v),abs=(v:F)=>v.n<BigInt(0)?neg(v):v,scale=[one,abs(a),abs(b)].reduce((s,v)=>le(s,v)?v:s,one);ok=le(a,b)&&le(mul(sub(b,a),f(BigInt(10)**BigInt(125))),scale);
   if(ok&&v.resolution!=='unresolved-at-precision-ceiling'){const mid=div(add(a,b),two),draw=binaryRatioObservation({coefficient:mid.n,exponent:0},{coefficient:mid.d,exponent:0});ok=v.value===draw.value&&(v.logMagnitude===null?draw.logMagnitude===null:typeof draw.logMagnitude==='number'&&Math.abs(v.logMagnitude-draw.logMagnitude)<=1e-12*Math.max(1,Math.abs(draw.logMagnitude)));}}
  cache.set(x,ok);return ok;
 };
 const exact:Check=x=>q(x)&&isMasterObservation(x,true),z:Check=x=>exact(x)&&(x as Q).sign===0,nonnegative:Check=x=>exact(x)&&le(zero,lo(x as Q)),positive:Check=x=>nonnegative(x)&&lo(x as Q).n>BigInt(0),nonnegativeInterval:Check=x=>q(x)&&le(zero,lo(x as Q)),prob:Check=x=>nonnegative(x)&&le(hi(x as Q),one),contains=(v:F):Check=>x=>q(x)&&le(lo(x as Q),v)&&le(v,hi(x as Q)),residual=contains(zero),complex=(c:Check=q):Check=>x=>shape(x,{real:c,imaginary:c}),matrix=(n:number,c:Check=q):Check=>array(n,array(n,complex(c))),rect=(a:number,b:number,c:Check=q):Check=>array(a,array(b,complex(c)));
 return{q,exact,z,nonnegative,positive,nonnegativeInterval,prob,contains,residual,complex,matrix,rect};
}
