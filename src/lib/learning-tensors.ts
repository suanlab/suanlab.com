import type { FoundationResult } from './learning-foundations';
import type { Vec2 } from './learning-simulations';
export const tensorKinds=['tensor-order-shapes','tensor-indices-contractions','multilinear-maps','broadcasting-semantics','tensor-cp-tucker','tensor-trains','tensor-networks-contraction','tensor-low-rank-approximation','tensor-symmetry-antisymmetry','tensor-storage-cost'] as const;
export type TensorKind=typeof tensorKinds[number];
export function isTensorKind(kind:string):kind is TensorKind{return (tensorKinds as readonly string[]).includes(kind);}
export interface TensorNetwork {nodes:{id:number;label:string;x:number;y:number}[];edges:{from:number;to:number;label:string}[];legs:{node:number;x:number;y:number;label:string}[];}
export interface TensorResult extends FoundationResult {table:{headers:string[];rows:(number|string)[][]};network?:TensorNetwork;}
function integer(x:number,min:number,max:number){if(!Number.isInteger(x)||x<min||x>max)throw new Error('Integer tensor parameter outside the example domain.');}
function range(x:number,min:number,max:number){if(x<min||x>max)throw new Error('Tensor parameter outside the example domain.');}
export function broadcastShape(a:number[],b:number[]):number[]|null{
 for(const shape of [a,b]){if(shape.length>8||!shape.every(x=>Number.isInteger(x)&&x>=0&&x<=1024))throw new Error('Invalid broadcasting shape.');}
 const result:number[]=[];for(let k=1;k<=Math.max(a.length,b.length);k++){const x=a[a.length-k]??1,y=b[b.length-k]??1;if(x===y)result.unshift(x);else if(x===1)result.unshift(y);else if(y===1)result.unshift(x);else return null;}return result;
}
export function matrixProduct(A:number[],B:number[],n:number,m:number,p:number){
 [n,m,p].forEach(x=>integer(x,1,4));if(A.length!==n*m||B.length!==m*p||![...A,...B].every(Number.isFinite))throw new Error('Matrix product shape mismatch.');let multiplies=0;
 const values=Array.from({length:n*p},(_,q)=>{const i=Math.floor(q/p),k=q%p;let sum=0;for(let j=0;j<m;j++){sum+=A[i*m+j]*B[j*p+k];multiplies++;}return sum;});return {values,multiplies};
}
export function contractCube(axis:number,v:number[]):number[]{integer(axis,0,2);if(v.length!==2||!v.every(Number.isFinite))throw new Error('Use a two-entry covector.');const free=[0,1,2].filter(a=>a!==axis);return Array.from({length:4},(_,q)=>{const index=[0,0,0];index[free[0]]=Math.floor(q/2);index[free[1]]=q%2;let sum=0;for(let k=0;k<2;k++){index[axis]=k;sum+=(1+4*index[0]+2*index[1]+index[2])*v[k];}return sum;});}
export function bilinearCube(u:number[],v:number[]):number[]{if(u.length!==2||v.length!==2||![...u,...v].every(Number.isFinite))throw new Error('Use two finite vectors of length two.');return [0,1].map(k=>{let sum=0;for(let i=0;i<2;i++)for(let j=0;j<2;j++)sum+=(1+4*i+2*j+k)*u[i]*v[j];return sum;});}
const indexed=(values:number[]):Vec2[]=>values.map((v,i)=>[i,v]);
const cubeRows=(original:number[],approx?:number[])=>original.map((v,q)=>[Math.floor(q/4),Math.floor(q/2)%2,q%2,v,...(approx?[approx[q]]:[])]);
function chainNetwork(labels:string[],bonds:string[],legs:{node:number;label:string}[]):TensorNetwork{return {nodes:labels.map((label,id)=>({id,label,x:55+120*id,y:75})),edges:bonds.map((label,i)=>({from:i,to:i+1,label})),legs:legs.map(leg=>({...leg,x:55+120*leg.node,y:170}))};}
export function evaluateTensor(kind:TensorKind,p:number[]):TensorResult{
 const counts:Record<TensorKind,number>={'tensor-order-shapes':4,'tensor-indices-contractions':3,'multilinear-maps':4,'broadcasting-semantics':3,'tensor-cp-tucker':3,'tensor-trains':3,'tensor-networks-contraction':4,'tensor-low-rank-approximation':3,'tensor-symmetry-antisymmetry':4,'tensor-storage-cost':4};
 if(p.length!==counts[kind]||!p.every(Number.isFinite))throw new Error('Use finite tensor inputs of the expected shape.');const [a,b,c,d]=p;
 if(kind==='tensor-order-shapes'){
  integer(a,1,3);integer(b,1,4);integer(c,1,4);integer(d,0,1);const shape=d===0?[a,b,c]:[a,c,b];const rows=Array.from({length:a*b*c},(_,q)=>{const i=Math.floor(q/(shape[1]*shape[2])),j=Math.floor(q/shape[2])%shape[1],k=q%shape[2];return [i,j,k,d===0?100*i+10*j+k:100*i+10*k+j];});const flat=rows.map(row=>row[3]);
  return {statement:`X[b,t,c]=100b+10t+c; input shape=(${a},${b},${c}); ${d===0?'identity axes':'permute axes (0,2,1)'}`,stats:{'array order':3,'input shape':[a,b,c],'output shape':shape,'stored entries':flat.length,'row-major output':flat},curves:[indexed(flat)],table:{headers:['axis 0','axis 1','axis 2','value'],rows}};
 }
 if(kind==='tensor-indices-contractions'){
  integer(a,0,2);integer(b,-2,2);integer(c,-2,2);const values=contractCube(a,[b,c]);
  return {statement:`X[i,j,k]=1+4i+2j+k; contract axis ${a} with v=(${b},${c})`,stats:{'free axes':[0,1,2].filter(x=>x!==a),'output shape':[2,2],'output row-major':values,'scalar multiplications':8,'scalar additions (two terms each)':4},curves:[indexed(values)],table:{headers:['free index 0','free index 1','contracted value'],rows:values.map((v,i)=>[Math.floor(i/2),i%2,v])}};
 }
 if(kind==='multilinear-maps'){
  p.forEach(x=>integer(x,-2,2));const u=[a,b],v=[c,d],f=bilinearCube(u,v),twice=bilinearCube(u.map(x=>2*x),v),both=bilinearCube(u.map(x=>2*x),v.map(x=>2*x)),sum=bilinearCube(u.map((x,i)=>x+v[i]),[1,0]),f1=bilinearCube(u,[1,0]),f2=bilinearCube(v,[1,0]);
  return {statement:`F(u,v)_k=Σᵢⱼ X[i,j,k]uᵢvⱼ; u=(${a},${b}), v=(${c},${d})`,stats:{'F(u,v)':f,'F(2u,v)':twice,'F(2u,2v)':both,'first-slot additivity residual':sum.map((x,i)=>x-f1[i]-f2[i])},curves:[indexed(f),indexed(twice)],table:{headers:['k','F(u,v)','F(2u,v)','F(2u,2v)'],rows:f.map((x,i)=>[i,x,twice[i],both[i]])}};
 }
 if(kind==='broadcasting-semantics'){
  integer(a,0,3);integer(b,-2,2);integer(c,0,1);const ashape=a===2?[2,1]:[2,3],bshape=a===0?[3]:a===1?[2,1]:a===2?[1,3]:[2],shape=broadcastShape(ashape,bshape);
  const values=shape?Array.from({length:6},(_,q)=>{const i=Math.floor(q/3),j=q%3,A=a===2?1+3*i:1+3*i+j,B=a===1?b*(i+1):b*(j+1);return c===0?A+B:A*B;}):null;
  return {statement:`A shape=(${ashape}); B shape=(${bshape}); op=${c===0?'elementwise +':'elementwise ×'}`,stats:{'A shape':ashape,'B shape':bshape,'1{broadcast compatible}':shape?1:0,'output shape':shape,'output row-major':values},curves:values?[indexed(values)]:[],table:{headers:['i','j','output'],rows:values?values.map((v,i)=>[Math.floor(i/3),i%3,v]):[]}};
 }
 if(kind==='tensor-cp-tucker'||kind==='tensor-low-rank-approximation'){
  integer(a,kind==='tensor-cp-tucker'?-2:-4,kind==='tensor-cp-tucker'?2:4);integer(b,kind==='tensor-cp-tucker'?-2:-4,kind==='tensor-cp-tucker'?2:4);integer(c,0,2);
  const weights=[a,b],chosen=[0,1].sort((i,j)=>Math.abs(weights[j])-Math.abs(weights[i])).slice(0,c).filter(i=>weights[i]!==0),r=chosen.length;
  const original=Array.from({length:8},(_,q)=>q===0?a:q===7?b:0),cp=Array.from({length:8},(_,q)=>chosen.reduce((s,i)=>s+(q===(i===0?0:7)?weights[i]:0),0));
  const tucker=Array.from({length:8},(_,q)=>{const indices=[Math.floor(q/4),Math.floor(q/2)%2,q%2];let sum=0;for(let u=0;u<r;u++)for(let v=0;v<r;v++)for(let w=0;w<r;w++){const core=u===v&&v===w?weights[chosen[u]]:0;sum+=core*Number(indices[0]===chosen[u])*Number(indices[1]===chosen[v])*Number(indices[2]===chosen[w]);}return sum;});
  const error=Math.hypot(...original.map((x,i)=>x-cp[i])),norm2=a*a+b*b;
  const stats:FoundationResult['stats']=kind==='tensor-cp-tucker'?{'retained nonzero components':r,'dense entries':8,'CP entries incl. weights':7*r,'Tucker entries':r**3+6*r,'CP–Tucker residual':Math.hypot(...cp.map((x,i)=>x-tucker[i])),'approximation error':error}:{'unfolding rank':Number(a!==0)+Number(b!==0),'retained components':r,'approximation error':error,'squared error':original.reduce((sum,x,i)=>sum+(x-cp[i])**2,0),'retained energy fraction':norm2===0?null:chosen.reduce((sum,i)=>sum+weights[i]**2,0)/norm2};
  const network:TensorNetwork={nodes:[{id:0,label:`G: ${r}³`,x:175,y:60},{id:1,label:`U₀: 2×${r}`,x:55,y:120},{id:2,label:`U₁: 2×${r}`,x:175,y:120},{id:3,label:`U₂: 2×${r}`,x:295,y:120}],edges:[1,2,3].map(to=>({from:0,to,label:`r=${r}`})),legs:[1,2,3].map((node,i)=>({node,x:55+120*i,y:195,label:`${['i','j','k'][i]}:2`}))};
  return {statement:`X=${a} e₀⊗e₀⊗e₀+${b} e₁⊗e₁⊗e₁; keep up to ${c} largest |weight| terms`,stats,curves:[indexed(original),indexed(cp)],network:kind==='tensor-cp-tucker'?network:undefined,table:{headers:['i','j','k','X','approx'],rows:cubeRows(original,cp)}};
 }
 if(kind==='tensor-trains'){
  integer(a,1,2);integer(b,-2,2);range(c,.25,2);const r=a,G0=Array.from({length:2*r},(_,q)=>Math.floor(q/r)===q%r?(q%r===0?c:1):0),G1=Array.from({length:r*2*r},(_,q)=>{const u=Math.floor(q/(2*r)),j=Math.floor(q/r)%2,v=q%r;return u===j&&j===v?(u===0?1/c:b):0;}),G2=Array.from({length:r*2},(_,q)=>Math.floor(q/2)===q%2?1:0);
  const values=Array.from({length:8},(_,q)=>{const i=Math.floor(q/4),j=Math.floor(q/2)%2,k=q%2;let sum=0;for(let u=0;u<r;u++)for(let v=0;v<r;v++)sum+=G0[i*r+u]*G1[(u*2+j)*r+v]*G2[v*2+k];return sum;});
  const target=Array.from({length:8},(_,q)=>q===0?1:q===7?b:0);
  return {statement:`X target=e₀⊗e₀⊗e₀+${b}e₁⊗e₁⊗e₁; configured bonds=${r}; gauge g=${c}; boundary bonds=1`,stats:{'core shapes (flattened)':[1,2,r,r,2,r,r,2,1],'stored core entries':G0.length+G1.length+G2.length,'core G₀ row-major':G0,'core G₁ row-major':G1,'core G₂ row-major':G2,'actual unfolding ranks':[1,a===2&&b!==0?2:1,a===2&&b!==0?2:1,1],'reconstruction error':Math.hypot(...target.map((x,i)=>x-values[i])),'reconstructed row-major':values},curves:[indexed(values)],network:chainNetwork(['G₀','G₁','G₂'],[`r=${r}`,`r=${r}`],[{node:0,label:'i:2'},{node:1,label:'j:2'},{node:2,label:'k:2'}]),table:{headers:['i','j','k','target','TT'],rows:cubeRows(target,values)}};
 }
 if(kind==='tensor-networks-contraction'){
  p.forEach(x=>integer(x,1,4));const A=Array.from({length:a*b},(_,q)=>Math.floor(q/b)+q%b+1),B=Array.from({length:b*c},(_,q)=>1+Math.floor(q/c)-q%c),C=Array.from({length:c*d},(_,q)=>Math.floor(q/d)+q%d+1),AB=matrixProduct(A,B,a,b,c),left=matrixProduct(AB.values,C,a,c,d),BC=matrixProduct(B,C,b,c,d),right=matrixProduct(A,BC.values,a,b,d);
  const network=chainNetwork([`A ${a}×${b}`,`B ${b}×${c}`,`C ${c}×${d}`],[`m=${b}`,`p=${c}`],[]);network.legs=[{node:0,x:10,y:145,label:`n=${a}`},{node:2,x:340,y:145,label:`q=${d}`}];
  return {statement:`A(${a},${b}) B(${b},${c}) C(${c},${d}); compare (AB)C and A(BC)`,stats:{'output shape':[a,d],'left scalar multiplications':AB.multiplies+left.multiplies,'right scalar multiplications':BC.multiplies+right.multiplies,'left intermediate entries':a*c,'right intermediate entries':b*d,'association residual':Math.hypot(...left.values.map((x,i)=>x-right.values[i]))},curves:[indexed(left.values)],network,table:{headers:['i','l','(AB)C','A(BC)'],rows:left.values.map((v,i)=>[Math.floor(i/d),i%d,v,right.values[i]])}};
 }
 if(kind==='tensor-symmetry-antisymmetry'){
  p.forEach(x=>integer(x,-3,3));const S=[a,(b+c)/2,(b+c)/2,d],K=[0,(b-c)/2,(c-b)/2,0];
  return {statement:`A=[${a},${b};${c},${d}]; S=(A+Aᵀ)/2; K=(A−Aᵀ)/2`,stats:{'symmetric part S':S,'antisymmetric part K':K,'reconstruction residual':Math.hypot(...p.map((v,i)=>v-S[i]-K[i])),'Frobenius inner product':S.reduce((sum,v,i)=>sum+v*K[i],0),'vᵀKv for v=(1,2)':K[0]+2*K[1]+2*K[2]+4*K[3]},curves:[indexed(S),indexed(K)],table:{headers:['row-major index','A','S','K'],rows:p.map((v,i)=>[i,v,S[i],K[i]])}};
 }
 integer(a,2,8);integer(b,2,8);integer(c,1,4);integer(d,0,1);const dense=b**a,tt=2*b*c+(a-2)*b*c*c,cp=c*(a*b+1),tucker=c**a+a*b*c,bytes=d===0?2:4;
 return {statement:`order d=${a}; each axis n=${b}; configured r=${c}; ${bytes} bytes/stored scalar; storage only`,stats:{'dense scalars':dense,'CP scalars incl. weights':cp,'Tucker scalars':tucker,'TT core scalars':tt,'dense bytes':dense*bytes,'TT bytes':tt*bytes,'dense/TT storage ratio':dense/tt},curves:[Array.from({length:7},(_,i)=>[i+2,Math.log10(b**(i+2))]),Array.from({length:7},(_,i)=>[i+2,Math.log10(2*b*c+i*b*c*c)])],table:{headers:['format','stored scalars','bytes'],rows:[['Dense',dense,dense*bytes],['CP + weights',cp,cp*bytes],['Tucker',tucker,tucker*bytes],['TT cores',tt,tt*bytes]]}};
}
