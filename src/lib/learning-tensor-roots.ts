/** Exact square-free factorization and Sturm certificates for real degree<=4 spectra.
 * No floating eigenvalues, root merging tolerance or rational-root enumeration.
 */
import {f,zero,one,add,sub,mul,div,neg,eq,type F}from'./learning-quantum-hilbert-core';
type P=F[];
const trim=(p:P):P=>{const r=p.slice();while(r.length>1&&r.at(-1)!.n===BigInt(0))r.pop();return r;},isZero=(p:P)=>p.length===1&&p[0].n===BigInt(0),degree=(p:P)=>isZero(p)?-1:p.length-1,monic=(p:P)=>{p=trim(p);if(isZero(p))return[zero];return p.map(x=>div(x,p.at(-1)!));};
function quotient(a:P,b:P){a=trim(a);b=trim(b);if(isZero(b))throw new RangeError('Nonzero polynomial divisor required');const q=Array.from({length:Math.max(1,a.length-b.length+1)},()=>zero);while(degree(a)>=degree(b)){const k=degree(a)-degree(b),s=div(a.at(-1)!,b.at(-1)!);q[k]=add(q[k],s);b.forEach((v,i)=>{a[i+k]=sub(a[i+k],mul(s,v));});a=trim(a);}return{q:trim(q),r:a};}
const derivative=(p:P)=>p.length<2?[zero]:p.slice(1).map((v,i)=>mul(f(BigInt(i+1)),v));
function gcd(a:P,b:P):P{while(!isZero(b)){const r=quotient(a,b).r;a=b;b=r;}return monic(a);}
function exactDivide(a:P,b:P){const r=quotient(a,b);if(!isZero(r.r))throw new RangeError('Exact polynomial factor required');return monic(r.q);}
const bgcd=(a:bigint,b:bigint):bigint=>{while(b!==BigInt(0)){const r=a%b;a=b;b=r;}return a<BigInt(0)?-a:a;};
function integerCoefficients(p:P){const lcm=p.reduce((d,v)=>d/bgcd(d,v.d)*v.d,BigInt(1));return p.map(v=>v.n*(lcm/v.d));}
function signAt(p:bigint[],n:bigint,d:bigint){let a=p.at(-1)!,power=d;for(let i=p.length-2;i>=0;i--){a=a*n+p[i]*power;power*=d;}return a===BigInt(0)?0:a>BigInt(0)?1:-1;}
function sturm(p:P){const rows=[p,derivative(p)];while(!isZero(rows.at(-1)!)){const r=quotient(rows.at(-2)!,rows.at(-1)!).r;if(isZero(r))break;rows.push(r.map(neg));}return rows.filter(v=>!isZero(v)).map(integerCoefficients);}
const variations=(rows:bigint[][],n:bigint,d:bigint)=>{const signs=rows.map(p=>signAt(p,n,d)).filter(v=>v!==0);return signs.slice(1).reduce((sum,v,i)=>sum+(v!==signs[i]?1:0),0);};
export type TensorIsolatedRoot={lower:F;upper:F;multiplicity:number;exact:boolean};
export function isolateTensorSpectrum(descending:F[],bits=1024):{roots:TensorIsolatedRoot[];squareFreeFactors:{multiplicity:number;ascending:F[]}[];isolationBits:number}{
 if(!Number.isSafeInteger(bits)||bits<512||bits>4096||descending.length<2||descending.length>5||descending.some(v=>typeof v.n!=='bigint'||typeof v.d!=='bigint'||v.d<=BigInt(0)||v.n.toString(2).length>16384||v.d.toString(2).length>16384)||descending[0].n===BigInt(0))throw new RangeError('Bounded nonzero polynomial degree1..4 and512..4096 isolation bits required');
 const p=monic(descending.slice().reverse());let g=gcd(p,derivative(p)),w=exactDivide(p,g),multiplicity=1;const factors:{multiplicity:number;ascending:F[]}[]=[];
 while(degree(w)>0){const y=gcd(w,g),z=exactDivide(w,y);if(degree(z)>0)factors.push({multiplicity,ascending:z});w=y;g=exactDivide(g,y);multiplicity++;if(multiplicity>5)throw new RangeError('Square-free factorization failed');}
 const target=BigInt(1)<<BigInt(bits),all:TensorIsolatedRoot[]=[];
 function isolate(p:P,m:number):TensorIsolatedRoot[]{
  if(degree(p)===0)return[];
  if(degree(p)===1){const r=neg(div(p[0],p[1]));return[{lower:r,upper:r,multiplicity:m,exact:true}];}
  const row=integerCoefficients(p),rows=sturm(p),bound=BigInt(2)+p.slice(0,-1).reduce((best,v)=>{const n=v.n<BigInt(0)?-v.n:v.n,t=(n+v.d-BigInt(1))/v.d;return t>best?t:best;},BigInt(0)),count=variations(rows,-bound,BigInt(1))-variations(rows,bound,BigInt(1));
  if(count!==degree(p))throw new RangeError('All spectral roots must be real');
  const found:TensorIsolatedRoot[]=[];let dyadic:F|null=null;
  // Keep the original left-first traversal,without one call frame per precision bit.
  const pending=[{lo:-bound,hi:bound,scale:BigInt(1),n:count,depth:0}];
  while(pending.length&&dyadic===null){
   const {lo,hi,scale,n,depth}=pending.pop()!;
   if(n===0)continue;if(depth>bits+256)throw new RangeError('Root isolation precision ceiling');
   if(n===1&&(hi-lo)*target<=scale){found.push({lower:f(lo,scale),upper:f(hi,scale),multiplicity:m,exact:false});continue;}
   const mid=lo+hi,nextScale=scale*BigInt(2);if(signAt(row,mid,nextScale)===0){dyadic=f(mid,nextScale);break;}
   const left=variations(rows,lo,scale)-variations(rows,mid,nextScale);
   if(n-left)pending.push({lo:mid,hi:hi*BigInt(2),scale:nextScale,n:n-left,depth:depth+1});
   if(left)pending.push({lo:lo*BigInt(2),hi:mid,scale:nextScale,n:left,depth:depth+1});
  }
  if(dyadic!==null){const root:F=dyadic;return[{lower:root,upper:root,multiplicity:m,exact:true},...isolate(exactDivide(p,[neg(root),one]),m)];}
  return found;
 }
 for(const v of factors)all.push(...isolate(v.ascending,v.multiplicity));all.sort((a,b)=>{const n=sub(a.lower,b.lower).n;return n===BigInt(0)?0:n<BigInt(0)?-1:1;});
 if(all.reduce((n,v)=>n+v.multiplicity,0)!==degree(p))throw new RangeError('Spectral multiplicities failed');for(let i=1;i<all.length;i++)if(sub(all[i].lower,all[i-1].upper).n<=BigInt(0)&&!eq(all[i].lower,all[i-1].upper))throw new RangeError('Distinct spectral intervals overlap');
 return{roots:all,squareFreeFactors:factors,isolationBits:bits};
}
