/** Shared response shape/observation protocol, without a producer model.
 * Caller must descriptor-snapshot wire values before using these helpers. */
import {f,one,add,sub,eq,le,observe,type F,type R}from'./learning-quantum-hilbert-core';
export class InformationResponseInvalid extends Error{constructor(readonly path:string,reason:string){super(reason);}}
export function informationRequire(condition:unknown,path:string,reason:string):asserts condition{if(!condition)throw new InformationResponseInvalid(path,reason);}
export function informationRecord(value:unknown,keys:readonly string[],path:string):Record<string,unknown>{
 informationRequire(value&&typeof value==='object'&&!Array.isArray(value),path,'Record required');const d=value as Record<string,unknown>;
 informationRequire(Object.keys(d).length===keys.length&&keys.every(k=>Object.hasOwn(d,k)),path,'Exact fields required');return d;
}
export function informationList(value:unknown,length:number,path:string):unknown[]{informationRequire(Array.isArray(value)&&value.length===length,path,'Exact array shape required');return value;}
export function informationSame(a:unknown,b:unknown):boolean{
 if(Object.is(a,b))return true;if(!a||!b||typeof a!=='object'||typeof b!=='object'||Array.isArray(a)!==Array.isArray(b))return false;
 const x=a as Record<string,unknown>,y=b as Record<string,unknown>,keys=Object.keys(x);return keys.length===Object.keys(y).length&&keys.every(k=>Object.hasOwn(y,k)&&informationSame(x[k],y[k]));
}
export type InformationEndpointBudget={coefficientDigits:number;binaryExponent:number};
const defaultBudget:InformationEndpointBudget={coefficientDigits:4096,binaryExponent:8192};
function endpointBudget(budget:InformationEndpointBudget){
 if(!Number.isSafeInteger(budget.coefficientDigits)||budget.coefficientDigits<1||budget.coefficientDigits>16384||!Number.isSafeInteger(budget.binaryExponent)||budget.binaryExponent<0||budget.binaryExponent>32768)throw new RangeError('Bounded endpoint budget required');
}
export function informationEndpoint(value:unknown,path:string,budget:InformationEndpointBudget=defaultBudget):F{
 endpointBudget(budget);
 const d=informationRecord(value,['numerator','denominator'],path);
 const binary=(v:unknown,name:string)=>{
  const x=informationRecord(v,['coefficient','exponent'],name);
  informationRequire(typeof x.coefficient==='string'&&x.coefficient.length<=budget.coefficientDigits&&/^(?:0|-?[1-9]\d*)$/.test(x.coefficient),name,'Bounded integer coefficient required');
  informationRequire(typeof x.exponent==='number'&&Number.isSafeInteger(x.exponent)&&Math.abs(x.exponent)<=budget.binaryExponent,name,'Bounded binary exponent required');return{n:BigInt(x.coefficient),e:x.exponent};
 };
 const n=binary(d.numerator,`${path}.numerator`),den=binary(d.denominator,`${path}.denominator`);informationRequire(den.n>BigInt(0),path,'Positive denominator required');
 const shift=n.e-den.e;return shift>=0?f(n.n<<BigInt(shift),den.n):f(n.n,den.n<<BigInt(-shift));
}
/** Read a previously owned observation; mathematical/metadata checks remain separate. */
export function informationObservationRange(value:unknown,path:string):R{
 const d=informationRecord(value,['value','sign','logMagnitude','logResolution','resolution','bounds'],path),b=informationRecord(d.bounds,['lower','upper'],`${path}.bounds`);
 return{lower:informationEndpoint(b.lower,`${path}.bounds.lower`),upper:informationEndpoint(b.upper,`${path}.bounds.upper`)};
}
export function createInformationObservationChecks(budget:InformationEndpointBudget=defaultBudget){
 endpointBudget(budget);
 let count=0;
 const check=(value:unknown,expected:R,path:string,exact=false)=>{
  const d=informationRecord(value,['value','sign','logMagnitude','logResolution','resolution','bounds'],path),bounds=informationRecord(d.bounds,['lower','upper'],`${path}.bounds`);
  const reported={lower:informationEndpoint(bounds.lower,`${path}.bounds.lower`,budget),upper:informationEndpoint(bounds.upper,`${path}.bounds.upper`,budget)};
  informationRequire(le(expected.lower,expected.upper),'$reference','Reversed independent certificate');
  informationRequire(le(reported.lower,reported.upper),path,'Reversed interval');
  informationRequire(le(reported.lower,expected.lower)&&le(expected.upper,reported.upper),path,'Independent certificate is outside reported interval');
  if(exact)informationRequire(eq(reported.lower,reported.upper),path,'Exact field requires equal endpoints');
  const abs=(x:F)=>f(x.n<BigInt(0)?-x.n:x.n,x.d),magnitude=le(abs(reported.lower),abs(reported.upper))?abs(reported.upper):abs(reported.lower);
  informationRequire(le(sub(reported.upper,reported.lower),f(add(one,magnitude).n,add(one,magnitude).d*(BigInt(1)<<BigInt(224)))),path,'Interval too wide');
  informationRequire(informationSame(value,observe(reported)),path,'Display metadata disagrees with bounds');count++;
 };
 return{check,exact:(value:unknown,expected:F,path:string)=>check(value,{lower:expected,upper:expected},path,true),get count(){return count;}};
}
