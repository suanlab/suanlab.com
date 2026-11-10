/** Exact Q(sqrt(2), i) matrices for bounded channel experiments. Worker/CLI only. */
import { f, zero, one, add, sub, mul, div, neg, eq, type F } from './learning-quantum-hilbert-core';
import { quadratic, circuitScalar as complex, circuitZero as Z, circuitOne as O, circuitAdd as plus,
  circuitMultiply as times, circuitConjugate as conjugate, circuitEqual as equal, circuitScalarWire,
  type Quadratic as Q, type CircuitScalar as C } from './learning-qubit-scalar';
export { f, zero, one, quadratic, complex, Z, O, plus, times, conjugate, equal, circuitScalarWire };
export type { F, Q, C };
export type M = C[][];
export const qAdd = (a: Q,b: Q): Q => [add(a[0],b[0]),add(a[1],b[1])];
export const qNeg = (a: Q): Q => [neg(a[0]),neg(a[1])];
export const qSub = (a: Q,b: Q) => qAdd(a,qNeg(b));
export const qMul = (a: Q,b: Q): Q => [add(mul(a[0],b[0]),mul(f(BigInt(2)),mul(a[1],b[1]))),add(mul(a[0],b[1]),mul(a[1],b[0]))];
export function qDiv(a: Q,b: Q): Q {
  const denominator=sub(mul(b[0],b[0]),mul(f(BigInt(2)),mul(b[1],b[1])));
  if(denominator.n===BigInt(0))throw new RangeError('Nonzero quadratic denominator required');
  const numerator=qMul(a,[b[0],neg(b[1])]);
  return [div(numerator[0],denominator),div(numerator[1],denominator)];
}
export const qEqual = (a: Q,b: Q) => eq(a[0],b[0])&&eq(a[1],b[1]);
const sign=(a:F)=>a.n===BigInt(0)?0:a.n>BigInt(0)?1:-1;
export function qSign(a:Q):number {
  const sa=sign(a[0]),sb=sign(a[1]);
  if(sb===0)return sa;if(sa===0||sa===sb)return sb;
  const delta=sign(sub(mul(a[0],a[0]),mul(f(BigInt(2)),mul(a[1],a[1]))));
  return delta===0?0:delta>0?sa:sb;
}
export const minus=(a:C,b:C)=>plus(a,complex(qNeg(b.re),qNeg(b.im)));
export const scale=(z:C,s:Q)=>complex(qMul(z.re,s),qMul(z.im,s));
export const divide=(z:C,s:Q)=>complex(qDiv(z.re,s),qDiv(z.im,s));
export const real=(n:number,d=1)=>complex(quadratic(f(BigInt(n),BigInt(d))));
export const zeros=(rows:number,columns=rows):M=>Array.from({length:rows},()=>Array.from({length:columns},()=>Z));
export const identity=(n:number):M=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?O:Z));
export const mAdd=(a:M,b:M):M=>a.map((row,i)=>row.map((z,j)=>plus(z,b[i][j])));
export const mSub=(a:M,b:M):M=>a.map((row,i)=>row.map((z,j)=>minus(z,b[i][j])));
export const mScale=(a:M,s:Q):M=>a.map(row=>row.map(z=>scale(z,s)));
export const adjoint=(a:M):M=>a[0].map((_,j)=>a.map(row=>conjugate(row[j])));
export const product=(a:M,b:M):M=>a.map(row=>b[0].map((_,j)=>row.reduce((s,z,k)=>plus(s,times(z,b[k][j])),Z)));
export const trace=(a:M):C=>a.reduce((s,row,i)=>plus(s,row[i]),Z);
export const mEqual=(a:M,b:M)=>a.length===b.length&&a.every((row,i)=>row.length===b[i].length&&row.every((z,j)=>equal(z,b[i][j])));
export const mWire=(a:M)=>a.map(row=>row.map(circuitScalarWire));
export const tensor=(a:M,b:M):M=>Array.from({length:a.length*b.length},(_,i)=>Array.from({length:a[0].length*b[0].length},(_,j)=>times(a[Math.floor(i/b.length)][Math.floor(j/b[0].length)],b[i%b.length][j%b[0].length])));

export function requireMatrix(a:M,rows:number,columns=rows){
  const pair=(x:Q)=>Array.isArray(x)&&x.length===2&&x.every(v=>v&&typeof v.n==='bigint'&&typeof v.d==='bigint'&&v.d>BigInt(0));
  if(!Array.isArray(a)||a.length!==rows||a.some(row=>!Array.isArray(row)||row.length!==columns||row.some(z=>!z||!pair(z.re)||!pair(z.im))))throw new RangeError('Invalid exact matrix shape or scalar');
}
export const isHermitian=(a:M)=>mEqual(a,adjoint(a));
export function determinant(a:M):C {
  if(a.length===1)return a[0][0];
  return a[0].reduce((s,z,j)=>{
    const minor=a.slice(1).map(row=>row.filter((_,k)=>k!==j));
    const term=times(z,determinant(minor));return j%2?minus(s,term):plus(s,term);
  },Z);
}
/** All principal minors, not just leading minors. Matrices bounded to dimension4. */
export function principalMinorCertificate(a:M){
  if(a.length<1||a.length>4)throw new RangeError('PSD certificate supports dimension one to four');
  requireMatrix(a,a.length);
  const hermitian=isHermitian(a);
  const minors=Array.from({length:2**a.length-1},(_,k)=>{
    const mask=k+1,indices=Array.from({length:a.length},(_,j)=>j).filter(j=>(mask&(1<<j))!==0);
    const value=determinant(indices.map(i=>indices.map(j=>a[i][j])));
    return {mask,value,nonnegative:qSign(value.re)>=0&&qEqual(value.im,quadratic(zero))};
  });
  return {hermitian,positiveSemidefinite:hermitian&&minors.every(m=>m.nonnegative),minors};
}
export function densityFromVector(z:C[]):M {
  if(z.length!==2)throw new RangeError('A qubit vector has two entries');
  requireMatrix(z.map(v=>[v]),2,1);
  const norm=z.reduce((s,v)=>qAdd(s,times(v,conjugate(v)).re),quadratic(zero));
  if(qSign(norm)<=0)throw new RangeError('Nonzero state required');
  return z.map(a=>z.map(b=>divide(times(a,conjugate(b)),norm)));
}
export function requireDensity(a:M,dimension=2){
  requireMatrix(a,dimension);
  if(!equal(trace(a),O)||!principalMinorCertificate(a).positiveSemidefinite)throw new RangeError('Positive unit-trace density matrix required');
}
/** Tensor order first factor then second factor. */
export function partialTrace(a:M,first:number,second:number,keep:'first'|'second'):M {
  if(!Number.isInteger(first)||!Number.isInteger(second)||first<1||second<1||first*second>8)throw new RangeError('Invalid bounded tensor dimensions');
  requireMatrix(a,first*second);
  if(keep==='first')return Array.from({length:first},(_,i)=>Array.from({length:first},(_,j)=>Array.from({length:second},(_,b)=>a[i*second+b][j*second+b]).reduce(plus,Z)));
  if(keep==='second')return Array.from({length:second},(_,i)=>Array.from({length:second},(_,j)=>Array.from({length:first},(_,b)=>a[b*second+i][b*second+j]).reduce(plus,Z)));
  throw new RangeError('Invalid retained tensor factor');
}
