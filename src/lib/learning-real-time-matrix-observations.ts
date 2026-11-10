/** Directed shared-scale observations of every symmetric matrix component.
 * Integer endpoints mean coefficient*2^(scalePowerOfTwo-coefficientBits).
 * Upper rows preserve all entries by ordinary symmetry (no conjugation).
 */
import type {F,C} from './learning-quantum-hilbert-core';
import {realTimeFloor,realTimeCeil} from './learning-real-time-interval';
const Z=BigInt(0),O=BigInt(1);
export interface BinaryCoefficientRange {lower:string;upper:string;}
const exponent=(values:F[])=>{
  let result:number|null=null;
  for(const x of values) {
    if(x.d<=Z)throw new RangeError('Matrix rational denominator must be positive');
    if(x.n===Z)continue;
    const magnitude=(x.n<Z?-x.n:x.n).toString(2).length-x.d.toString(2).length+1;
    if(result===null||magnitude>result)result=magnitude;
  }
  return result??0;
};
const observe=(x:F,e:number):BinaryCoefficientRange=>{
  if(x.n===Z)return{lower:'0',upper:'0'};
  const shift=512-e,n=shift>=0?x.n<<BigInt(shift):x.n,d=shift>=0?x.d:x.d<<BigInt(-shift);
  return{lower:String(realTimeFloor(n,d)-O),upper:String(realTimeCeil(n,d)+O)};
};
function dimension(d:number){if(!Number.isInteger(d)||d<1||d>31)throw new RangeError('Expected1..31interior coordinates');}
export function packRealTimeSymmetricReal(d:number,entry:(i:number,j:number)=>F) {
  dimension(d);const rows=Array.from({length:d},(_,i)=>Array.from({length:d-i},(_,k)=>entry(i,i+k)));
  const e=exponent(rows.flat());
  return{encoding:'symmetric-real-binary-intervals' as const,dimension:d,coefficientBits:512,scalePowerOfTwo:e,upperRows:rows.map(row=>row.map(x=>observe(x,e)))};
}
export function packRealTimeSymmetricComplex(d:number,entry:(i:number,j:number)=>C) {
  dimension(d);const rows=Array.from({length:d},(_,i)=>Array.from({length:d-i},(_,k)=>entry(i,i+k)));
  const e=exponent(rows.flatMap(row=>row.flatMap(x=>[x.re,x.im])));
  return{encoding:'symmetric-complex-binary-intervals' as const,dimension:d,coefficientBits:512,scalePowerOfTwo:e,upperRows:rows.map(row=>row.map(x=>({real:observe(x.re,e),imaginary:observe(x.im,e)})))};
}
