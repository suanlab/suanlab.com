/** Exact sums/products of supplied binary64 inputs, followed by one projection.
 * This does not restore information already lost in geometric input calculations.
 */
export interface BinaryValue { coefficient: bigint; exponent: number }
export interface BinaryObservation { value: number|null; sign:number; logMagnitude:number|null }
const ZERO=BigInt(0),ONE=BigInt(1),TWO=BigInt(2),view=new DataView(new ArrayBuffer(8));
export const binaryZero:BinaryValue={coefficient:ZERO,exponent:0};
function fromNumber(value:number):BinaryValue {
  if(!Number.isFinite(value))throw new RangeError('Expected finite binary input');
  if(value===0)return binaryZero;
  view.setFloat64(0,value);const bits=view.getBigUint64(0),fraction=bits&((ONE<<BigInt(52))-ONE),stored=Number((bits>>BigInt(52))&BigInt(2047));
  return{coefficient:(value<0?-ONE:ONE)*(stored===0?fraction:fraction+(ONE<<BigInt(52))),exponent:stored===0?-1074:stored-1023-52};
}
export function binaryProduct(...values:number[]):BinaryValue {
  let coefficient=ONE,exponent=0;
  for(const value of values){const b=fromNumber(value);coefficient*=b.coefficient;exponent+=b.exponent;}
  return coefficient===ZERO?binaryZero:{coefficient,exponent};
}
export function binarySum(values:BinaryValue[]):BinaryValue {
  const active=values.filter(v=>v.coefficient!==ZERO);if(!active.length)return binaryZero;
  const exponent=Math.min(...active.map(v=>v.exponent));
  return{coefficient:active.reduce((sum,v)=>sum+(v.coefficient<<BigInt(v.exponent-exponent)),ZERO),exponent};
}
export const binaryNegate=(v:BinaryValue):BinaryValue=>({coefficient:-v.coefficient,exponent:v.exponent});
export function binaryMultiply(a:BinaryValue,b:BinaryValue):BinaryValue {
  return{coefficient:a.coefficient*b.coefficient,exponent:a.exponent+b.exponent};
}
const bits=(n:bigint)=>n.toString(2).length;
function logarithm(n:bigint){const shift=Math.max(0,bits(n)-53);return Math.log(Number(n>>BigInt(shift)))+shift*Math.LN2;}
/** Exact rational rounding to nearest binary64, ties to even; null for a nonzero
 * result outside the finite nonzero linear range. The logarithm is approximate.
 */
export function binaryRatioObservation(a:BinaryValue,b:BinaryValue):BinaryObservation {
  if(b.coefficient===ZERO)throw new RangeError('Zero binary denominator');
  if(a.coefficient===ZERO)return{value:0,sign:0,logMagnitude:null};
  const sign=(a.coefficient<ZERO?-1:1)*(b.coefficient<ZERO?-1:1);
  const numerator=a.coefficient<ZERO?-a.coefficient:a.coefficient,denominator=b.coefficient<ZERO?-b.coefficient:b.coefficient;
  const exp=a.exponent-b.exponent,d=bits(numerator)-bits(denominator);
  const below=d>=0?numerator<(denominator<<BigInt(d)):(numerator<<BigInt(-d))<denominator;
  const highest=d+exp-(below?1:0),quantum=Math.max(-1074,highest-52),shift=exp-quantum;
  const n=shift>=0?numerator<<BigInt(shift):numerator,den=shift<0?denominator<<BigInt(-shift):denominator;
  let mantissa=n/den;const remainder=n%den;
  if(TWO*remainder>den||TWO*remainder===den&&mantissa%TWO!==ZERO)mantissa+=ONE;
  const projected=sign*Number(mantissa)*2**quantum;
  const value=projected===0||!Number.isFinite(projected)?null:projected;
  return{value,sign,logMagnitude:logarithm(numerator)-logarithm(denominator)+exp*Math.LN2};
}
export const binaryObservation=(value:BinaryValue)=>binaryRatioObservation(value,{coefficient:ONE,exponent:0});
