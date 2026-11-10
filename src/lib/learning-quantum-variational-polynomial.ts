/** Degree<=4 exact factor/multiplicity accounts,without numerical root search. */
import{f,zero,add,sub,mul,div,eq,le,lo,hi,type F}from'./learning-tensor-wire-math';
import{rationalGreatestCommonDivisor as bgcd}from'./learning-rational-gcd';
import type{PartitionObservation as Q}from'./learning-partition-exact-core';
type P=F[];const trim=(p:P)=>{const a=p.slice();while(a.length>1&&a.at(-1)!.n===BigInt(0))a.pop();return a;},isZero=(p:P)=>p.length===1&&p[0].n===BigInt(0),degree=(p:P)=>isZero(p)?-1:p.length-1,monic=(p:P)=>isZero(p)?[zero]:p.map(v=>div(v,p.at(-1)!));
function quotient(a:P,b:P){a=trim(a);b=trim(b);if(isZero(b))throw new RangeError('Nonzero polynomial divisor');const q=Array.from({length:Math.max(1,a.length-b.length+1)},()=>zero);while(degree(a)>=degree(b)){const k=degree(a)-degree(b),s=div(a.at(-1)!,b.at(-1)!);q[k]=add(q[k],s);b.forEach((v,i)=>{a[k+i]=sub(a[k+i],mul(s,v));});a=trim(a);}return{q:trim(q),r:a};}
const derivative=(p:P)=>p.length<2?[zero]:p.slice(1).map((v,i)=>mul(v,f(BigInt(i+1))));
function gcd(a:P,b:P):P{while(!isZero(b)){const r=quotient(a,b).r;a=b;b=r;}return monic(a);}
const divide=(a:P,b:P)=>{const v=quotient(a,b);if(!isZero(v.r))throw new RangeError('Exact polynomial factor');return monic(v.q);};
export function variationalSquareFree(descending:P){const p=monic(descending.slice().reverse());let g=gcd(p,derivative(p)),w=divide(p,g),m=1;const factors:{multiplicity:number;ascending:P}[]=[];while(degree(w)>0){const y=gcd(w,g),z=divide(w,y);if(degree(z)>0)factors.push({multiplicity:m,ascending:z});w=y;g=divide(g,y);if(++m>5)throw new RangeError('Bounded multiplicity');}return factors;}
function coefficients(p:P){const den=p.reduce((d,v)=>d/bgcd(d,v.d)*v.d,BigInt(1));return p.map(v=>v.n*(den/v.d));}
function sign(p:bigint[],x:F){let a=p.at(-1)!,d=x.d;for(let i=p.length-2;i>=0;i--){a=a*x.n+p[i]*d;d*=x.d;}return a===BigInt(0)?0:a>BigInt(0)?1:-1;}
export function certifiedPrimarySpectrum(rows:{energy:Q;lower:Q;upper:Q;multiplicity:number;exact:boolean}[],descending:P,bits=1024){
 const factors=variationalSquareFree(descending),used=new Map<number,number>();if(rows.reduce((n,v)=>n+v.multiplicity,0)!==descending.length-1)return false;
 for(let i=0;i<rows.length;i++){
  const row=rows[i],a=lo(row.lower),b=lo(row.upper),factor=factors.find(v=>v.multiplicity===row.multiplicity);if(!factor||!le(a,b)||!le(mul(sub(b,a),f(BigInt(1)<<BigInt(bits))),f(BigInt(1)))||row.exact!==eq(a,b)||!le(lo(row.energy),a)||!le(b,hi(row.energy)))return false;
  if(i>0&&(!le(lo(rows[i-1].upper),a)||eq(lo(rows[i-1].lower),b)))return false;
  const p=coefficients(factor.ascending),sa=sign(p,a),sb=sign(p,b);if(row.exact?sa!==0:sa*sb>=0)return false;used.set(row.multiplicity,(used.get(row.multiplicity)??0)+1);
 }
 return factors.every(v=>used.get(v.multiplicity)===v.ascending.length-1);
}
function sturm(p:P){const rows=[p,derivative(p)];while(!isZero(rows.at(-1)!)){const r=quotient(rows.at(-2)!,rows.at(-1)!).r;if(isZero(r))break;rows.push(r.map(v=>({n:-v.n,d:v.d})));}return rows.filter(v=>!isZero(v)).map(coefficients);}
function variations(signs:number[]){const s=signs.filter(v=>v!==0);return s.slice(1).reduce((n,v,i)=>n+(v!==s[i]?1:0),0);}
function counts(p:P,a:F,b:F){const rows=sturm(p),left=variations(rows.map(v=>sign(v,a))),right=variations(rows.map(v=>sign(v,b))),infinity=variations(rows.map(v=>{const lead=v.at(-1)!;return(lead>BigInt(0)?1:-1)*((v.length-1)%2?-1:1);})),pointA=sign(coefficients(p),a)===0?1:0;return{below:infinity-left-pointA,closed:left-right+pointA};}
export function certifiedMinimum(row:{energy:Q;lower:Q;upper:Q;multiplicity:number;exact:boolean},descending:P,bits=4096){
 const factors=variationalSquareFree(descending),a=lo(row.lower),b=lo(row.upper);if(!le(a,b)||!le(mul(sub(b,a),f(BigInt(1)<<BigInt(bits))),f(BigInt(1)))||row.exact!==eq(a,b)||!le(lo(row.energy),a)||!le(b,hi(row.energy)))return false;
 let matched=false;for(const factor of factors){const n=counts(factor.ascending,a,b);if(n.below!==0)return false;if(factor.multiplicity===row.multiplicity){if(n.closed!==1)return false;matched=true;}else if(n.closed!==0)return false;}return matched;
}
