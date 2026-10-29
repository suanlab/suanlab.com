import type { FoundationResult } from './learning-foundations';
import { symmetricEigen2 } from './learning-linear-algebra';
import type { Vec2 } from './learning-simulations';
export const matrixStructureKinds=['vector-spaces-rank','determinants-traces','positive-semidefinite-matrices','quadratic-forms-orders','kronecker-products'] as const;
export type MatrixStructureKind=typeof matrixStructureKinds[number];
export function isMatrixStructureKind(kind:string):kind is MatrixStructureKind{return (matrixStructureKinds as readonly string[]).includes(kind);}
export interface MatrixStructureResult extends FoundationResult {table?:{headers:string[];rows:(number|string)[][]};}
/** Integers keep the rank and PSD predicates exact in these deliberately small examples. */
export function rank2(a:number,b:number,c:number,d:number){return a*d-b*c!==0?2:[a,b,c,d].some(x=>x!==0)?1:0;}
export function psd2(a:number,b:number,c:number){return a>=0&&c>=0&&a*c-b*b>=0;}
export function kronecker2(A:number[],B:number[]){if(A.length!==4||B.length!==4||![...A,...B].every(Number.isFinite))throw new Error('Use finite 2×2 matrices.');return Array.from({length:16},(_,i)=>{const r=Math.floor(i/4),c=i%4;return A[Math.floor(r/2)*2+Math.floor(c/2)]*B[(r%2)*2+c%2];});}
export function evaluateMatrixStructure(kind:MatrixStructureKind,p:number[]):MatrixStructureResult{
 const count:Record<MatrixStructureKind,number>={'vector-spaces-rank':4,'determinants-traces':4,'positive-semidefinite-matrices':3,'quadratic-forms-orders':5,'kronecker-products':2};
 if(p.length!==count[kind]||!p.every(x=>Number.isInteger(x)&&x>=-360&&x<=360))throw new Error('Use small finite integer parameters of the expected shape.');
 const [a,b,c,d,e]=p;
 if(kind==='vector-spaces-rank'){
  const rank=rank2(a,b,c,d),basis=rank===2?[0,1]:rank===1?(a!==0||c!==0?[0]:[1]):[];
  const kernel=rank===1?(a!==0||b!==0?[-b,a]:[-d,c]):null;
  return {statement:`A=[${a},${b};${c},${d}]; columns v₀,v₁`,stats:{'rank(A)':rank,'dim ker(A)':2-rank,'column-basis indices':basis,'kernel generator (rank 1)':kernel,'kernel residual A·v':kernel?[a*kernel[0]+b*kernel[1],c*kernel[0]+d*kernel[1]]:null,'1{columns independent}':rank===2?1:0},curves:[[[0,0],[a,c]],[[0,0],[b,d]]]};
 }
 if(kind==='determinants-traces'){
  const det=a*d-b*c,trace=a+d,disc=trace*trace-4*det;const roots=disc>=0?[(trace+Math.sqrt(disc))/2,(trace-Math.sqrt(disc))/2]:null;
  return {statement:`A=[${a},${b};${c},${d}]; oriented columns`,stats:{'det A':det,'area |det A|':Math.abs(det),'tr A':trace,'characteristic discriminant':disc,'real eigenvalues':roots,'complex pair Re, |Im|':disc<0?[trace/2,Math.sqrt(-disc)/2]:null},curves:[[[0,0],[a,c],[a+b,c+d],[b,d],[0,0]],[[0,0],[1,0],[1,1],[0,1],[0,0]]]};
 }
 if(kind==='positive-semidefinite-matrices'){
  const [hi,lo]=symmetricEigen2(a,b,c).values,positive=psd2(a,b,c),definite=a>0&&a*c-b*b>0;
  return {statement:`A=[${a},${b};${b},${c}]; q(v)=vᵀAv; ‖v‖₂=1`,stats:{'λmax':hi,'λmin':lo,'det A':a*c-b*b,'1{A⪰0}':positive?1:0,'1{A≻0}':definite?1:0,'rank(A)':rank2(a,b,b,c)},curves:[Array.from({length:181},(_,i)=>{const theta=i*2*Math.PI/180;return [i*2,a*Math.cos(theta)**2+2*b*Math.cos(theta)*Math.sin(theta)+c*Math.sin(theta)**2] as Vec2;}),[[0,0],[360,0]]]};
 }
 if(kind==='quadratic-forms-orders'){
  const [hi,lo]=symmetricEigen2(a,b,c).values,theta=d*Math.PI/180,v=[Math.cos(theta),Math.sin(theta)],q=a*v[0]**2+2*b*v[0]*v[1]+c*v[1]**2;
  return {statement:`A=[${a},${b};${b},${c}]; v=(cosθ,sinθ), θ=${d}°; candidate C=${e}`,stats:{'vᵀAv':q,'λmin':lo,'λmax':hi,'candidate C':e,'1{A⪯CI}':psd2(e-a,-b,e-c)?1:0,'1{probe vᵀAv≤C}':q<=e+1e-12?1:0},curves:[Array.from({length:181},(_,i)=>{const t=i*2*Math.PI/180;return [i*2,a*Math.cos(t)**2+2*b*Math.cos(t)*Math.sin(t)+c*Math.sin(t)**2] as Vec2;}),[[0,e],[360,e]]]};
 }
 const A=[1,a,0,1],B=[b,0,0,1],X=[1,3,2,4],AB=kronecker2(A,B),BA=kronecker2(B,A),multiply=(M:number[])=>Array.from({length:4},(_,i)=>X.reduce((s,v,j)=>s+M[i*4+j]*v,0)),correct=[b*(1+3*a),3*b,2+4*a,4],viaKronecker=multiply(BA),wrong=multiply(AB);
 return {statement:`A=[1,${a};0,1]; B=[${b},0;0,1]; X=[1,2;3,4]; column-major vec`,stats:{'vec(X)':X,'vec(AXBᵀ)':correct,'(A⊗B)vec(X)':wrong,'‖vec(AXBᵀ)−(B⊗A)vec(X)‖₂':Math.hypot(...correct.map((v,i)=>v-viaKronecker[i])),'rank(A⊗B)':b===0?2:4,'det(A⊗B)':b*b},curves:[correct.map((v,i)=>[i,v]),wrong.map((v,i)=>[i,v])],table:{headers:['row','col 0','col 1','col 2','col 3'],rows:Array.from({length:4},(_,i)=>[i,...AB.slice(i*4,i*4+4)])}};
}
