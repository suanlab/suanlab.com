import type { FoundationResult } from './learning-foundations';
import type { DiscreteNetwork } from './learning-discrete';
export const abstractAlgebraKinds=['groups-rings-fields','homomorphisms-isomorphisms','subgroups-quotients','polynomials-finite-fields','modules-linear-actions','algebraic-symmetries','polynomial-systems','invariant-constructions','algebraic-representations-scope'] as const;
export type AbstractAlgebraKind=typeof abstractAlgebraKinds[number];
export const isAbstractAlgebraKind=(k:string):k is AbstractAlgebraKind=>(abstractAlgebraKinds as readonly string[]).includes(k);
export interface AbstractAlgebraResult extends FoundationResult {table:{headers:string[];rows:(number|string)[][]};network?:DiscreteNetwork;}
export const mod=(x:number,n:number)=>((x%n)+n)%n;
export function gcd(a:number,b:number):number{while(b){const r=a%b;a=b;b=r;}return Math.abs(a);}
export function residueInverse(a:number,n:number):number|null{if(!Number.isInteger(n)||n<2||n>12||!Number.isInteger(a))throw new Error('Invalid residue domain.');for(let b=0;b<n;b++)if(mod(a*b,n)===1)return b;return null;}
export function binaryPolynomialProduct(a:number,b:number,m:number):number{if(![a,b].every(x=>Number.isInteger(x)&&x>=0&&x<8)||![10,11].includes(m))throw new Error('Invalid binary polynomial representation.');let value=0;for(let j=0;j<3;j++)if((b>>j)&1)value^=a<<j;for(let j=4;j>=3;j--)if((value>>j)&1)value^=m<<(j-3);return value;}
export function squareAction(r:number,f:number,v:number[]):number[]{if(!Number.isInteger(r)||r<0||r>3||![0,1].includes(f)||v.length!==2||!v.every(Number.isFinite))throw new Error('Invalid square action.');let [x,y]=v;if(f)y=-y;for(let k=0;k<r;k++)[x,y]=[-y,x];return [x===0?0:x,y===0?0:y];}
const indexed=(a:number[]):[number,number][]=>a.map((x,i)=>[i,x]);
function orbitNetwork(points:number[][],r:number,f:number):DiscreteNetwork{return {directed:true,nodes:points.map(([x,y],id)=>({id,label:`(${x},${y})`,x:175+85*x/Math.max(1,...points.flat().map(Math.abs)),y:150-85*y/Math.max(1,...points.flat().map(Math.abs))})),edges:points.map((p,from)=>{const q=squareAction(r,f,p);return {from,to:points.findIndex(v=>v[0]===q[0]&&v[1]===q[1]),active:true};})};}
export function evaluateAbstractAlgebra(kind:AbstractAlgebraKind,p:number[]):AbstractAlgebraResult{
 const bounds:Record<AbstractAlgebraKind,[number,number][]>={'groups-rings-fields':[[2,9],[0,8]],'homomorphisms-isomorphisms':[[2,8],[2,8],[0,7]],'subgroups-quotients':[[2,9],[0,8]],'polynomials-finite-fields':[[0,1],[0,7],[0,7]],'modules-linear-actions':[[2,7],[0,6],[0,6]],'algebraic-symmetries':[[0,3],[0,1],[0,2]],'polynomial-systems':[[0,1],[0,6],[0,6]],'invariant-constructions':[[-2,2],[-2,2],[0,2]],'algebraic-representations-scope':[[-2,2],[-2,2],[-2,2],[-2,2]]};
 if(p.length!==bounds[kind].length||p.some((x,i)=>!Number.isInteger(x)||x<bounds[kind][i][0]||x>bounds[kind][i][1]))throw new Error('Use integer parameters in the stated algebra domain.');const [a,b,c,d]=p;
 if(kind==='groups-rings-fields'){
  const x=mod(b,a),units=Array.from({length:a},(_,i)=>i).filter(i=>residueInverse(i,a)!==null),prime=units.length===a-1;
  return {statement:`Z/${a}Z; a=${x}; + and × are modulo ${a}`,stats:{'normalized a':x,'additive inverse':mod(-x,a),'multiplicative inverse':residueInverse(x,a),'units':units,'1{field}':Number(prime)},curves:[indexed(Array.from({length:a},(_,y)=>mod(x*y,a)))],table:{headers:['y','a+y','a×y','inverse(y)'],rows:Array.from({length:a},(_,y)=>[y,mod(x+y,a),mod(x*y,a),residueInverse(y,a)??'—'])}};
 }
 if(kind==='homomorphisms-isomorphisms'){
  const k=mod(c,b),valid=mod(k*a,b)===0,values=Array.from({length:a},(_,x)=>mod(k*x,b)),image=[...new Set(values)].sort((x,y)=>x-y),kernel=values.flatMap((v,i)=>v===0?[i]:[]);
  return {statement:`Candidate f: Z/${a}Z → Z/${b}Z, [x] ↦ [${k}x]; representatives 0…${a-1}`,stats:{'representative consistency residual':mod(k*a,b),'1{homomorphism}':Number(valid),'kernel':valid?kernel:null,'image':valid?image:null,'1{isomorphism}':valid?Number(kernel.length===1&&image.length===b):null},curves:[indexed(values)],table:{headers:['representative x','candidate kx mod n'],rows:values.map((v,i)=>[i,v])}};
 }
 if(kind==='subgroups-quotients'){
  const h=mod(b,a),g=gcd(a,h),H=Array.from({length:a/g},(_,k)=>mod(k*h,a)).sort((x,y)=>x-y);
  return {statement:`H=⟨${h}⟩ ≤ Z/${a}Z; coset label x mod ${g}`,stats:{'subgroup H':H,'subgroup size':H.length,'coset count':g,'|H| × |G/H|':H.length*g},curves:[indexed(Array.from({length:a},(_,x)=>x%g))],table:{headers:['x','coset label','x+H (sorted)'],rows:Array.from({length:a},(_,x)=>[x,x%g,H.map(v=>mod(x+v,a)).sort((u,v)=>u-v).join(', ')])}};
 }
 if(kind==='polynomials-finite-fields'){
  const m=a===0?11:10,mul=(u:number,v:number)=>binaryPolynomialProduct(u,v,m),inverse=(u:number)=>{for(let v=0;v<8;v++)if(mul(u,v)===1)return v;return null;};
  return {statement:`F₂[t]/(${a===0?'t³+t+1':'t³+t'}); integer bits encode polynomial coefficients`,stats:{'a+b (XOR)':b^c,'a×b reduced':mul(b,c),'inverse(a)':inverse(b),'unit count':Array.from({length:8},(_,i)=>i).filter(i=>inverse(i)!==null).length,'1{field}':Number(a===0)},curves:[indexed(Array.from({length:8},(_,v)=>mul(b,v)))],table:{headers:['b bits as integer','a×b reduced','inverse(b)'],rows:Array.from({length:8},(_,v)=>[v,mul(b,v),inverse(v)??'—'])}};
 }
 if(kind==='modules-linear-actions'){
  const aa=mod(b,a),bb=mod(c,a),rows=Array.from({length:a*a},(_,i)=>{const x=Math.floor(i/a),y=i%a;return [x,y,mod(aa*x,a),mod(bb*y,a)];}),ker=rows.filter(r=>r[2]===0&&r[3]===0).length;
  return {statement:`M=(Z/${a}Z)²; T(x,y)=(${aa}x,${bb}y) mod ${a}`,stats:{'kernel size':ker,'image size':a*a/ker,'1{invertible module map}':Number(ker===1),'field dimension of kernel':Array.from({length:a-1},(_,i)=>i+1).every(i=>gcd(i,a)===1)?(ker===1?0:ker===a?1:2):null},curves:[indexed(rows.map(r=>r[2]*a+r[3]))],table:{headers:['x','y','T₀','T₁'],rows}};
 }
 if(kind==='algebraic-symmetries'){
  const v=c===0?[1,1]:c===1?[1,0]:[0,0],orbit:number[][]=[],all=Array.from({length:8},(_,k)=>squareAction(k%4,Math.floor(k/4),v));all.forEach(q=>{if(!orbit.some(x=>x[0]===q[0]&&x[1]===q[1]))orbit.push(q);});const target=squareAction(a,b,v),stab=all.filter(x=>x[0]===v[0]&&x[1]===v[1]).length;
  return {statement:`D₄ action R^${a} S^${b}; apply S first, R second; R(x,y)=(-y,x), S(x,y)=(x,-y)`,stats:{'transformed point':target,'orbit size':orbit.length,'stabilizer size':stab,'orbit × stabilizer':orbit.length*stab},curves:[indexed(target)],network:orbitNetwork(orbit,a,b),table:{headers:['r','f','RʳSᶠ v: x','y'],rows:all.map((q,k)=>[k%4,Math.floor(k/4),...q])}};
 }
 if(kind==='polynomial-systems'){
  const n=a===0?5:7,s=mod(b,n),product=mod(c,n),rows=Array.from({length:n*n},(_,i)=>{const x=Math.floor(i/n),y=i%n;return [x,y,mod(x+y-s,n),mod(x*y-product,n)];}),solutions=rows.filter(r=>r[2]===0&&r[3]===0);
  return {statement:`Over F_${n}: x+y=${s}, xy=${product}; exhaustive grid, ordered solutions`,stats:{'solution count':solutions.length,'solutions flattened':solutions.flatMap(r=>r.slice(0,2)),'discriminant mod p':mod(s*s-4*product,n)},curves:[indexed(rows.map(r=>Number(r[2]===0&&r[3]===0)))],table:{headers:['x','y','x+y−s mod p','xy−q mod p'],rows}};
 }
 if(kind==='invariant-constructions'){
  const orbit=Array.from({length:4},(_,k)=>squareAction(k,0,[a,b])),feature=(x:number,y:number)=>c===0?x:c===1?x*x:x*y,values=orbit.map(([x,y])=>feature(x,y)),avg=values.reduce((s,v)=>s+v,0)/4;
  return {statement:`C₄ orbit of (${a},${b}); φ=${c===0?'x':c===1?'x²':'xy'}; Pφ(v)=¼Σₖφ(Rᵏv)`,stats:{'orbit feature values':values,'averaged invariant':avg,'norm squared':a*a+b*b,'mean x':orbit.reduce((s,v)=>s+v[0],0)/4,'mean y':orbit.reduce((s,v)=>s+v[1],0)/4},curves:[indexed(values),indexed(values.map(()=>avg))],table:{headers:['k','x','y','φ(Rᵏv)','Pφ'],rows:orbit.map((v,i)=>[i,...v,values[i],avg])}};
 }
 const M=[a,b,c,d],J=[0,-1,1,0],mul=(u:number[],v:number[])=>[u[0]*v[0]+u[1]*v[2],u[0]*v[1]+u[1]*v[3],u[2]*v[0]+u[3]*v[2],u[2]*v[1]+u[3]*v[3]],comm=(u:number[],v:number[])=>Math.hypot(...mul(u,v).map((x,i)=>x-mul(v,u)[i]));
 let projected=[0,0,0,0];for(let k=0;k<4;k++){const R=squareAction(k,0,[1,0]),S=squareAction(k,0,[0,1]),r=[R[0],S[0],R[1],S[1]],inv=[r[0],r[2],r[1],r[3]],term=mul(mul(inv,M),r);projected=projected.map((x,i)=>x+term[i]/4);}
 return {statement:'Real C₄ representation ρ(R)=J; project M by ¼Σₖ J⁻ᵏ M Jᵏ; compare with D₄ reflection S',stats:{'original C₄ commutator norm':comm(M,J),'C₄ projected row-major':projected,'projected C₄ residual':comm(projected,J),'projected reflection residual':comm(projected,[1,0,0,-1]),'D₄ projected row-major':[(a+d)/2,0,0,(a+d)/2]},curves:[indexed(M),indexed(projected)],table:{headers:['row-major index','M','C₄ projection'],rows:M.map((x,i)=>[i,x,projected[i]])}};
}
