import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateMatrixStructure as evaluate, matrixStructureKinds, rank2, psd2, kronecker2 } from '../src/lib/learning-matrix-structures';
import { matrixStructureLabs, learningGuides, learningUnits } from '../src/data/interactive-book';
function mul(A:number[],B:number[],n:number){return Array.from({length:n*n},(_,i)=>Array.from({length:n},(_,k)=>A[Math.floor(i/n)*n+k]*B[k*n+i%n]).reduce((s,v)=>s+v,0));}
function det(A:number[],n:number):number{if(n===1)return A[0];return Array.from({length:n},(_,j)=>(j%2?-1:1)*A[j]*det(A.filter((_,i)=>Math.floor(i/n)>0&&i%n!==j),n-1)).reduce((s,v)=>s+v,0);}
const clean=(xs:number[])=>xs.map(v=>v===0?0:v);
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('rank, column bases and null generators cover all small integer two-dimensional maps',()=>{
 for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)for(let d=-2;d<=2;d++){
  const r=evaluate('vector-spaces-rank',[a,b,c,d]),rank=r.stats['rank(A)'] as number,indices=r.stats['column-basis indices'] as number[];assert.equal(rank+(r.stats['dim ker(A)'] as number),2);assert.equal(indices.length,rank);
  if(rank===1){const v=r.stats['kernel generator (rank 1)'] as number[];assert.ok(v.some(x=>x!==0));assert.ok(a*v[0]+b*v[1]===0);assert.ok(c*v[0]+d*v[1]===0);const j=indices[0];assert.ok([a,b][j]!==0||[c,d][j]!==0);}
  if(rank===0)assert.ok([a,b,c,d].every(x=>x===0));if(rank===2)assert.notEqual(det([a,b,c,d],2),0);
 }
});
test('determinants match oriented polygon area; trace and determinant survive similarity',()=>{
 const P=[1,1,0,1],inverse=[1,-1,0,1];
 for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)for(let d=-2;d<=2;d++){
  const A=[a,b,c,d],r=evaluate('determinants-traces',A),path=r.curves[0];let area=0;for(let i=0;i<path.length-1;i++)area+=path[i][0]*path[i+1][1]-path[i+1][0]*path[i][1];close(area/2,r.stats['det A'] as number);
  const B=mul(mul(inverse,A,2),P,2);assert.equal(B[0]+B[3],r.stats['tr A']);close(det(B,2),r.stats['det A'] as number);
 }
 assert.deepEqual(evaluate('determinants-traces',[0,-1,1,0]).stats['complex pair Re, |Im|'],[0,1]);
});
test('PSD certificates agree with eigenvalues, including zero, singular and indefinite boundaries',()=>{
 for(let a=-3;a<=3;a++)for(let b=-3;b<=3;b++)for(let c=-3;c<=3;c++){
  const r=evaluate('positive-semidefinite-matrices',[a,b,c]);assert.equal(psd2(a,b,c),(r.stats['λmin'] as number)>=-1e-12);assert.equal(r.stats['1{A⪰0}'],Number(psd2(a,b,c)));assert.equal(r.stats['1{A≻0}'],Number((r.stats['λmin'] as number)>1e-12));assert.equal(r.stats['rank(A)'],rank2(a,b,b,c));
  for(const [,q]of r.curves[0]){assert.ok(q>=(r.stats['λmin'] as number)-1e-12);assert.ok(q<=(r.stats['λmax'] as number)+1e-12);}
 }
 assert.equal(evaluate('positive-semidefinite-matrices',[0,0,0]).stats['1{A⪰0}'],1);assert.equal(evaluate('positive-semidefinite-matrices',[-1,0,-1]).stats['1{A⪰0}'],0);
});
test('Loewner certificates are universal while a passing probe can fail to certify them',()=>{
 for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)for(let C=-3;C<=4;C++){
  const r=evaluate('quadratic-forms-orders',[a,b,c,0,C]);assert.equal(r.stats['1{A⪯CI}'],Number(C>=(r.stats['λmax'] as number)-1e-12));
 }
 const r=evaluate('quadratic-forms-orders',[2,1,2,0,2]);assert.equal(r.stats['1{probe vᵀAv≤C}'],1);assert.equal(r.stats['1{A⪯CI}'],0);const counterexample=evaluate('quadratic-forms-orders',[2,1,2,45,2]);close(counterexample.stats['vᵀAv'] as number,3);assert.equal(counterexample.stats['1{probe vᵀAv≤C}'],0);
});
test('Kronecker mixed-product identity is independently checked by ordinary matrix multiplication',()=>{
 const matrices=[[1,2,3,4],[0,1,-1,2],[2,0,0,0],[1,-1,0,1]];
 for(const A of matrices)for(const B of matrices)for(const C of matrices)for(const D of matrices)assert.deepEqual(clean(mul(kronecker2(A,B),kronecker2(C,D),4)),clean(kronecker2(mul(A,C,2),mul(B,D,2))));
});
test('column vectorization matches direct two-sided multiplication and exposes reversed factors',()=>{
 for(let s=-3;s<=3;s++)for(let t=-3;t<=3;t++){
  const r=evaluate('kronecker-products',[s,t]),direct=mul(mul([1,s,0,1],[1,2,3,4],2),[t,0,0,1],2),col=[direct[0],direct[2],direct[1],direct[3]];assert.deepEqual(clean(r.stats['vec(AXBᵀ)'] as number[]),clean(col));assert.equal(r.stats['‖vec(AXBᵀ)−(B⊗A)vec(X)‖₂'],0);assert.equal(det(kronecker2([1,s,0,1],[t,0,0,1]),4),r.stats['det(A⊗B)']);assert.equal(r.stats['rank(A⊗B)'],t===0?2:4);
 }
 const r=evaluate('kronecker-products',[1,2]);assert.deepEqual(r.stats['vec(AXBᵀ)'],[8,6,6,4]);assert.deepEqual(r.stats['(A⊗B)vec(X)'],[6,7,4,4]);
});
for(const kind of matrixStructureKinds)test(`matrix-structure guides and finite preset geometry: ${kind}`,()=>{
 const spec=matrixStructureLabs[kind];assert.ok(learningUnits.some(u=>u.id===kind));assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);for(const preset of spec.presets){preset.values.forEach((x,i)=>assert.ok(x>=spec.controls[i].min&&x<=spec.controls[i].max));const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));}
});
test('matrix examples reject malformed or fractional exact-arithmetic inputs',()=>{
 for(const kind of matrixStructureKinds){assert.throws(()=>evaluate(kind,[]));const values=matrixStructureLabs[kind].controls.map(c=>c.initial);values[0]=.1;assert.throws(()=>evaluate(kind,values));values[0]=Infinity;assert.throws(()=>evaluate(kind,values));}assert.throws(()=>kronecker2([1,2],[1,2,3,4]));
});
