/** Bounded owned data and observation schemas;no global precision-width assumption. */
import{quantumVariationalRecord as record}from'./learning-quantum-variational-input';
import{isMasterObservation}from'./learning-master-equation-protocol';
import{binaryRatioObservation}from'./learning-binary-products';
import{one,two,zero,add,div,le,lo,hi,type F}from'./learning-tensor-wire-math';
import type{PartitionObservation as Q}from'./learning-partition-exact-core';
export type Check=(x:unknown)=>boolean;
export function shape(x:unknown,s:Record<string,Check>):x is Record<string,unknown>{
 if(!record(x)||Reflect.ownKeys(x).length!==Object.keys(s).length)return false;const ds=Object.getOwnPropertyDescriptors(x);return Object.entries(s).every(([k,c])=>Object.hasOwn(ds,k)&&Object.hasOwn(ds[k],'value')&&c(ds[k].value));
}
export const literal=(v:unknown):Check=>x=>x===v,choice=(...v:unknown[]):Check=>x=>v.includes(x),bool:Check=x=>typeof x==='boolean',integer=(x:unknown,min=0,max=Number.MAX_SAFE_INTEGER):x is number=>typeof x==='number'&&Number.isSafeInteger(x)&&x>=min&&x<=max;
export const array=(n:number,c:Check):Check=>x=>{if(!Array.isArray(x)||Object.getPrototypeOf(x)!==Array.prototype||x.length!==n||Reflect.ownKeys(x).length!==n+1)return false;const ds=Object.getOwnPropertyDescriptors(x);return Array.from({length:n},(_,i)=>Object.hasOwn(ds,i)&&Object.hasOwn(ds[i],'value')&&c(ds[i].value)).every(Boolean);};
export const list=(min:number,max:number,c:Check):Check=>x=>Array.isArray(x)&&x.length>=min&&x.length<=max&&array(x.length,c)(x),nullable=(c:Check):Check=>x=>x===null||c(x),fields=(names:string,c:Check)=>Object.fromEntries(names.split(' ').map(k=>[k,c]));
export function createQuantumVariationalWireChecks(){
 const cache=new WeakMap<object,boolean>(),term=(positive=false):Check=>x=>shape(x,{coefficient:v=>typeof v==='string'&&v.length<=8192&&(positive?/^[1-9]\d*$/:/^-?(?:0|[1-9]\d*)$/).test(v)&&v!=='-0',exponent:literal(0)}),fraction:Check=x=>shape(x,{numerator:term(),denominator:term(true)});
 const q:Check=x=>{
  if(!record(x))return false;if(cache.has(x))return cache.get(x)!;
  let ok=shape(x,{value:v=>v===null||typeof v==='number'&&Number.isFinite(v),sign:choice(-1,0,1,null),logMagnitude:v=>v===null||typeof v==='number'&&Number.isFinite(v),logResolution:choice('finite','not-applicable','unresolved'),resolution:choice('structural-zero','exact-rational','interval-resolved','unresolved-at-precision-ceiling'),bounds:v=>shape(v,{lower:fraction,upper:fraction})})&&isMasterObservation(x);
  if(ok){const v=x as unknown as Q,a=lo(v),b=hi(v);ok=le(a,b);if(ok&&v.resolution!=='unresolved-at-precision-ceiling'){const mid=div(add(a,b),two),draw=binaryRatioObservation({coefficient:mid.n,exponent:0},{coefficient:mid.d,exponent:0});ok=v.value===draw.value&&(v.logMagnitude===null?draw.logMagnitude===null:typeof draw.logMagnitude==='number'&&Math.abs(v.logMagnitude-draw.logMagnitude)<=1e-12*Math.max(1,Math.abs(draw.logMagnitude)));}}
  cache.set(x,ok);return ok;
 };
 const exact:Check=x=>q(x)&&isMasterObservation(x,true),z:Check=x=>exact(x)&&(x as Q).sign===0,positive:Check=x=>exact(x)&&lo(x as Q).n>BigInt(0),nonnegative:Check=x=>exact(x)&&le(zero,lo(x as Q)),nonnegativeInterval:Check=x=>q(x)&&le(zero,lo(x as Q)),probability:Check=x=>nonnegativeInterval(x)&&le(hi(x as Q),one),contains=(v:F):Check=>x=>q(x)&&le(lo(x as Q),v)&&le(v,hi(x as Q)),residual=contains(zero),complex=(c:Check=q):Check=>x=>shape(x,{real:c,imaginary:c}),vector=(n:number,c:Check=q):Check=>array(n,complex(c)),rect=(n:number,m:number,c:Check=q):Check=>array(n,array(m,complex(c))),matrix=(n:number,c:Check=q)=>rect(n,n,c);
 return{q,exact,z,positive,nonnegative,nonnegativeInterval,probability,contains,residual,complex,vector,rect,matrix};
}
