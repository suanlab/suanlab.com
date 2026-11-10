/** Independent proof-checking arithmetic. No producer engine/interval imports.
 * Interval endpoints are integers on a 1280-bit grid; exact proof witnesses use
 * the separate Rational implementation. All candidate parsing is bounded first.
 */
import { Rational as Q, Z, O } from './learning-graph-gauge-validation-math';
export { Q, Z, O };
export const BITS = 1280, UNIT = BigInt(1), NIL = BigInt(0), SCALE = UNIT << BigInt(BITS);
export type Band = { lo: bigint; hi: bigint };
export const integer = (n: number) => new Q(BigInt(n));
export const compare = (a: Q, b: Q) => a.n*b.d < b.n*a.d ? -1 : a.n*b.d > b.n*a.d ? 1 : 0;
export const abs = (x: Q) => x.n < NIL ? x.neg() : x;
export const max = (a: Q, b: Q) => compare(a,b) > 0 ? a : b;
export const min = (a: Q, b: Q) => compare(a,b) < 0 ? a : b;
export const square = (x: Q) => x.mul(x);
export const sum = (xs: Q[]) => xs.reduce((a,b)=>a.add(b), Z);
export const floor = (n: bigint, d: bigint): bigint => n >= NIL ? n/d : -((-n+d-UNIT)/d);
export const ceil = (n: bigint, d: bigint) => -floor(-n,d);
export const exact = (x: Q): Band => ({lo: floor(x.n*SCALE,x.d), hi: ceil(x.n*SCALE,x.d)});
export const interval = (lo: Q, hi: Q): Band => {if(compare(lo,hi)>0)throw Error('Unordered exact proof interval');return {lo:exact(lo).lo,hi:exact(hi).hi};};
export const zero: Band = {lo:NIL,hi:NIL}, one: Band = {lo:SCALE,hi:SCALE};
export const add = (a: Band, b: Band): Band => ({lo:a.lo+b.lo,hi:a.hi+b.hi});
export const neg = (a: Band): Band => ({lo:-a.hi,hi:-a.lo});
export const sub = (a: Band, b: Band) => add(a,neg(b));
const least = (xs: bigint[]) => xs.reduce((a,b)=>a<b?a:b);
const greatest = (xs: bigint[]) => xs.reduce((a,b)=>a>b?a:b);
export const mul = (a: Band, b: Band): Band => { const xs=[a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi]; return {lo:floor(least(xs),SCALE),hi:ceil(greatest(xs),SCALE)}; };
export const div = (a: Band, b: Band): Band => {
  if(b.lo<=NIL&&b.hi>=NIL) throw Error('Unresolved proof denominator');
  const ns=[a.lo,a.lo,a.hi,a.hi],ds=[b.lo,b.hi,b.lo,b.hi];
  return {lo:least(ns.map((n,i)=>floor((ds[i]<NIL?-n:n)*SCALE,ds[i]<NIL?-ds[i]:ds[i]))),hi:greatest(ns.map((n,i)=>ceil((ds[i]<NIL?-n:n)*SCALE,ds[i]<NIL?-ds[i]:ds[i])))};
};
export const sq = (a: Band): Band => ({lo:a.lo<=NIL&&a.hi>=NIL?NIL:floor(least([a.lo*a.lo,a.hi*a.hi]),SCALE),hi:ceil(greatest([a.lo*a.lo,a.hi*a.hi]),SCALE)});
export const scale = (a: Band, x: Q) => mul(a,exact(x));
export const total = (xs: Band[]) => xs.reduce(add,zero);
export const positive = (a: Band): Band => {if(a.hi<NIL||a.lo>a.hi)throw Error('Nonnegative proof range missed');return {lo:a.lo<NIL?NIL:a.lo,hi:a.hi};};
export const negative = (a: Band): Band => {if(a.lo>NIL||a.lo>a.hi)throw Error('Nonpositive proof range missed');return {lo:a.lo,hi:a.hi>NIL?NIL:a.hi};};
export const rational = (v: bigint) => new Q(v,SCALE);
/** Continuant Green formula, independent of the producer's LDL column solves. */
export function green(diagonal: Band[]) {
  const n=diagonal.length,left=[one],right:Band[]=Array(n+1);right[n]=one;
  for(let i=0;i<n;i++)left.push(sub(mul(diagonal[i],left[i]),i?left[i-1]:zero));
  for(let i=n-1;i>=0;i--)right[i]=sub(mul(diagonal[i],right[i+1]),i<n-1?right[i+2]:zero);
  if(left[n].lo<=NIL)throw Error('Unresolved positive continuant');
  const inverse=diagonal.map((_,i)=>diagonal.map((__,j)=>positive(div(mul(left[Math.min(i,j)],right[Math.max(i,j)+1]),left[n]))));
  return {inverse,determinant:left[n],pivots:diagonal.map((_,i)=>div(left[i+1],left[i]))};
}
function integerRoot(n: bigint, power: number): bigint {
  if(n<NIL)throw Error('Negative root');if(n< BigInt(2))return n;
  let x=UNIT<<BigInt(Math.ceil(n.toString(2).length/power));
  const p=BigInt(power),previous=BigInt(power-1);
  for(;;){const y=(previous*x+n/(x**previous))/p;if(y>=x)return x;x=y;}
}
export function root(x: Q, power: 2|4): Band {
  const quotient=x.n*(SCALE**BigInt(power))/x.d,lo=integerRoot(quotient,power);
  return {lo,hi:lo**BigInt(power)*x.d===x.n*(SCALE**BigInt(power))?lo:lo+UNIT};
}
/** Canonical wire rationals only. Length checks happen before BigInt or gcd. */
export function readBounded(value: unknown, maximumLength=4096): Q {
  if(typeof value!=='string'||value.length>maximumLength||! /^(?:0|-?[1-9][0-9]*)(?:\/[1-9][0-9]*)?$/.test(value))throw Error('Invalid bounded rational');
  const x=Q.read(value);if(x.text()!==value)throw Error('Noncanonical rational');return x;
}
/** Semantic enclosure checks, not byte equality with one solver's rounding.
 * Both ends must enclose this independently certified reference. A relative /
 * absolute 2^-480 width ceiling prevents vacuous mathematical enclosures.
 */
export function verifyObservation(value: unknown, reference: Band): boolean {
  if(reference.lo>reference.hi)return false;
  if(!value||typeof value!=='object'||Array.isArray(value))return false;
  const v=value as Record<string,unknown>;
  if(Object.keys(v).length!==2||!Object.hasOwn(v,'lower')||!Object.hasOwn(v,'upper'))return false;
  const lo=readBounded(v.lower),hi=readBounded(v.upper),denominator=UNIT<<BigInt(512);
  if(lo.d>denominator||hi.d>denominator||(lo.d&(lo.d-UNIT))!==NIL||(hi.d&(hi.d-UNIT))!==NIL)return false;
  const a=rational(reference.lo),b=rational(reference.hi);
  const budget=max(O,max(abs(a),abs(b))).div(new Q(UNIT<<BigInt(480)));
  return compare(lo,a)<=0&&compare(hi,b)>=0&&compare(hi.sub(lo),budget)<=0;
}
