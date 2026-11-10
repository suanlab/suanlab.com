/** Worker-side exact quadratic data. No positivity/pivot assumptions, phase
 * observations, Airy evaluation or probability interpretation are imposed.
 * Full Green matrices are accessible from lossless continuant factors.
 */
import {readRealTimePhaseInput,type RealTimeChainInput} from './learning-real-time-phase-input';
import {f,zero,one,two,add,sub,mul,div,neg,sum,c,ca,cs,cm,scaleC,abs2,type F,type C} from './learning-quantum-hilbert-core';
const Z=BigInt(0),integer=(n:number)=>f(BigInt(n)),square=(x:F)=>mul(x,x);
export const realTimeFraction=(value:string):F=>{const[n,d='1']=value.split('/');return f(BigInt(n),BigInt(d));};
export const realTimeFractionText=(value:F)=>value.d===BigInt(1)?String(value.n):`${value.n}/${value.d}`;
const complexDivide=(a:C,b:C)=>scaleC(cm(a,c(b.re,neg(b.im))),div(one,abs2(b)));
export type RealTimeExactMatrix=F[][];
export function realTimeMatrixProduct(a:RealTimeExactMatrix,b:RealTimeExactMatrix):RealTimeExactMatrix {
  return a.map(row=>b[0].map((_,j)=>sum(row.map((v,k)=>mul(v,b[k][j])))));
}
const transpose=(a:RealTimeExactMatrix)=>a[0].map((_,j)=>a.map(row=>row[j]));
function power(a:RealTimeExactMatrix,n:number):RealTimeExactMatrix {
  let out=[[one,zero],[zero,one]];
  while(n){if(n%2)out=realTimeMatrixProduct(out,a);n=Math.floor(n/2);if(n)a=realTimeMatrixProduct(a,a);}
  return out;
}
export function realTimeExactMap(intervals:number,time:F,mass:F,omega:F) {
  const delta=div(time,integer(intervals)),v=mul(omega,square(delta)),diagonal=sub(one,div(v,two));
  const step=[[diagonal,div(delta,mass)],[neg(mul(mul(mul(mass,omega),delta),sub(one,div(v,integer(4))))),diagonal]];
  return {delta,v,step,total:power(step,intervals)};
}
/** Exact finite state, with access to every H/Green entry. */
export function buildRealTimeChainExact(value:unknown) {
  const input=readRealTimePhaseInput(value);
  if(!input||input.mode!=='chain')throw new RangeError('Real-time chain input required');
  const n=input.intervals,d=n-1,time=realTimeFraction(input.time),mass=realTimeFraction(input.mass),hbar=realTimeFraction(input.hbar),omega=realTimeFraction(input.omegaSquared),start=realTimeFraction(input.start),end=realTimeFraction(input.end),damping=realTimeFraction(input.damping);
  const map=realTimeExactMap(n,time,mass,omega),s=div(mass,map.delta),diagonal=mul(s,sub(two,map.v));
  const theta=[one,sub(two,map.v)];for(let k=2;k<=d;k++)theta.push(sub(mul(theta[1],theta[k-1]),theta[k-2]));
  const signs=theta.filter(x=>x.n!==Z).map(x=>x.n>Z?1:-1);
  const negative=signs.slice(1).filter((x,i)=>x!==signs[i]).length,nullity=theta[d].n===Z?1:0;
  const determinant=mul(theta[d],f(s.n**BigInt(d),s.d**BigInt(d)));
  const forcing=Array.from({length:d},()=>zero);forcing[0]=add(forcing[0],mul(s,start));forcing[d-1]=add(forcing[d-1],mul(s,end));
  const endpoints=add(square(start),square(end));
  const constant=sub(div(mul(s,endpoints),two),div(mul(mul(mul(map.delta,mass),omega),endpoints),integer(4)));
  const hessianEntry=(i:number,j:number)=>i===j?diagonal:Math.abs(i-j)===1?neg(s):zero;
  const inverseEntry=(i:number,j:number):F=>{
    if(nullity)throw new RangeError('Singular Hessian has no inverse');
    return div(mul(theta[Math.min(i,j)],theta[d-1-Math.max(i,j)]),mul(s,theta[d]));
  };
  let stationary:F[]|null=null,stationaryAction:F|null=null;
  const nullVector=nullity?theta.slice(0,d):null;
  const compatible=!nullVector||sum(nullVector.map((x,i)=>mul(x,forcing[i]))).n===Z;
  if(!nullity) {
    stationary=Array.from({length:d},(_,i)=>sum(forcing.flatMap((x,j)=>x.n===Z?[]:[mul(inverseEntry(i,j),x)])));
    stationaryAction=sub(constant,div(sum(forcing.map((x,i)=>mul(x,stationary![i]))),two));
  } else if(compatible) {
    // Choose x1=0. Recurrence determines a representative; adding t*r
    // gives the entire affine family. No fictitious inverse is returned.
    const path=[start,zero];for(let i=1;i<n;i++)path.push(sub(mul(theta[1],path[i]),path[i-1]));
    stationary=path.slice(1,n);stationaryAction=zero;
    if(sub(path[n],end).n!==Z)throw Error('Caustic representative failed endpoint compatibility');
  }
  let regulator:ReturnType<typeof regulated>|null=null;
  if(damping.n!==Z)regulator=regulated(d,s,diagonal,damping,forcing,hbar,constant);
  return {input,n,d,time,mass,hbar,omega,start,end,damping,s,diagonal,theta,determinant,forcing,constant,
    inertia:{positive:d-negative-nullity,negative,zero:nullity},nullVector,compatible,
    stationaryStatus:nullity?(compatible?'affine-family':'no-stationary-path'):'unique',stationary,stationaryAction,
    causticIndex:nullity?negative+1:null,map,hessianEntry,inverseEntry,regulator,
    refinements:[1,2,4,8].map(multiplier=>({multiplier,intervals:n*multiplier,map:multiplier===1?map.total:realTimeExactMap(n*multiplier,time,mass,omega).total}))};
}
function regulated(d:number,s:F,diagonal:F,damping:F,forcing:F[],hbar:F,constant:F) {
  const a=c(damping,neg(diagonal)),theta=[c(one),a],s2=square(s);
  for(let k=2;k<=d;k++)theta.push(ca(cm(a,theta[k-1]),scaleC(theta[k-2],s2)));
  const pivots=Array.from({length:d},(_,i)=>complexDivide(theta[i+1],theta[i]));
  if(pivots.some(x=>x.re.n<=Z))throw Error('Abel pivot must have positive real part');
  const distancePowers=[c(one)];for(let k=1;k<d;k++)distancePowers.push(cm(distancePowers[k-1],c(zero,neg(s))));
  const inverseEntry=(i:number,j:number):C=>complexDivide(cm(cm(theta[Math.min(i,j)],theta[d-1-Math.max(i,j)]),distancePowers[Math.abs(i-j)]),theta[d]);
  const inverseForcing=Array.from({length:d},(_,i)=>forcing.reduce((out,x,j)=>x.n===Z?out:ca(out,scaleC(inverseEntry(i,j),x)),c(zero)));
  const mean=inverseForcing.map(x=>cm(c(zero,neg(one)),x));
  const covarianceEntry=(i:number,j:number)=>scaleC(inverseEntry(i,j),hbar);
  const exponent=cs(c(zero,div(constant,hbar)),scaleC(forcing.reduce((out,x,i)=>ca(out,scaleC(inverseForcing[i],x)),c(zero)),div(one,mul(two,hbar))));
  return {theta,pivots,distancePowers,determinant:theta[d],inverseEntry,inverseForcing,mean,covarianceEntry,exponent};
}
export function buildRealTimePacketExact(value:unknown) {
  const input=readRealTimePhaseInput(value);if(!input||input.mode!=='packet')throw new RangeError('Real-time packet input required');
  const n=input.intervals,time=realTimeFraction(input.time),mass=realTimeFraction(input.mass),hbar=realTimeFraction(input.hbar),omega=realTimeFraction(input.omegaSquared),position=realTimeFraction(input.position),momentum=realTimeFraction(input.momentum),chirp=realTimeFraction(input.chirp),width=realTimeFraction(input.width);
  const variance=input.widthMode==='fixed-variance'?width:mul(hbar,width),map=realTimeExactMap(n,time,mass,omega),M=map.total;
  const initialCovariance=[[variance,mul(chirp,variance)],[mul(chirp,variance),add(mul(square(chirp),variance),div(square(hbar),mul(integer(4),variance)))]];
  const finalCovariance=realTimeMatrixProduct(realTimeMatrixProduct(M,initialCovariance),transpose(M));
  const center=M.map(row=>add(mul(row[0],position),mul(row[1],momentum)));
  const centralAction=div(sub(mul(center[0],center[1]),mul(position,momentum)),two);
  const initialWidth=c(chirp,div(hbar,mul(two,variance)));
  const finalWidth=complexDivide(ca(c(M[1][0]),scaleC(initialWidth,M[1][1])),ca(c(M[0][0]),scaleC(initialWidth,M[0][1])));
  if(finalWidth.im.n<=Z)throw Error('Propagated Gaussian width must have positive imaginary part');
  // Reuse the chain index so the eventual metaplectic prefactor follows the
  // same finite split branch; the squared prefactor alone is insufficient.
  const chain:RealTimeChainInput={mode:'chain',intervals:n,time:input.time,mass:input.mass,hbar:input.hbar,omegaSquared:input.omegaSquared,start:input.position,end:'0',damping:'0',probe:0};
  const branch=buildRealTimeChainExact(chain);
  return {input,map,variance,initialCovariance,finalCovariance,center,centralAction,initialWidth,finalWidth,
    negativeModes:branch.inertia.negative,causticIndex:branch.causticIndex,
    refinements:branch.refinements,hbar,chirp,position,momentum};
}
