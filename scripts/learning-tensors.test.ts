import test from 'node:test';
import assert from 'node:assert/strict';
import { tensorKinds, evaluateTensor as evaluate, broadcastShape, contractCube, bilinearCube, matrixProduct } from '../src/lib/learning-tensors';
import { tensorLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('axis permutations reorder coordinates without changing entries or tensor order',()=>{
 for(let B=1;B<=3;B++)for(let T=1;T<=4;T++)for(let C=1;C<=4;C++){
  const x=evaluate('tensor-order-shapes',[B,T,C,0]),y=evaluate('tensor-order-shapes',[B,T,C,1]);
  assert.equal(y.stats['array order'],3);assert.deepEqual(y.stats['output shape'],[B,C,T]);assert.deepEqual([...x.stats['row-major output'] as number[]].sort((a,b)=>a-b),[...y.stats['row-major output'] as number[]].sort((a,b)=>a-b));
  y.table.rows.forEach(([b,c,t,v])=>assert.equal(v,100*Number(b)+10*Number(t)+Number(c)));
 }
 assert.deepEqual(evaluate('tensor-order-shapes',[1,2,3,1]).stats['row-major output'],[0,10,1,11,2,12]);
});
test('contraction agrees with an independent enumeration and preserves ordered free axes',()=>{
 for(let axis=0;axis<3;axis++)for(const v of [[1,2],[0,0],[-2,1]]){
  const result=[0,0,0,0],free=[0,1,2].filter(a=>a!==axis);
  for(let i=0;i<2;i++)for(let j=0;j<2;j++)for(let k=0;k<2;k++){const idx=[i,j,k];result[2*idx[free[0]]+idx[free[1]]]+=(1+4*i+2*j+k)*v[idx[axis]];}
  assert.deepEqual(contractCube(axis,v),result);assert.deepEqual(evaluate('tensor-indices-contractions',[axis,...v]).stats['free axes'],free);
 }
});
test('bilinear evaluation is linear in each slot but scales quadratically when both change',()=>{
 for(const u of [[1,0],[-2,1],[0,0]])for(const v of [[0,1],[1,1],[-1,2]]){
  const f=bilinearCube(u,v),scaled=bilinearCube(u.map(x=>2*x),v.map(x=>2*x));assert.deepEqual(scaled,f.map(x=>4*x));
  const w=[1,-1],sum=bilinearCube(u.map((x,i)=>x+w[i]),v),other=bilinearCube(w,v);assert.deepEqual(sum,f.map((x,i)=>x+other[i]));
 }
 assert.deepEqual(bilinearCube([1,1],[1,1]),[16,20]);
});
test('broadcasting aligns trailing axes, preserves zero dimensions and refuses incompatible output',()=>{
 assert.deepEqual(broadcastShape([5,1],[1,6]),[5,6]);assert.deepEqual(broadcastShape([0,3],[1,3]),[0,3]);assert.deepEqual(broadcastShape([],[]),[]);assert.deepEqual(broadcastShape([],[2,3]),[2,3]);assert.equal(broadcastShape([2,3],[2]),null);
 assert.deepEqual(evaluate('broadcasting-semantics',[0,1,0]).stats['output row-major'],[2,4,6,5,7,9]);assert.deepEqual(evaluate('broadcasting-semantics',[1,1,0]).stats['output row-major'],[2,3,4,6,7,8]);assert.deepEqual(evaluate('broadcasting-semantics',[2,1,1]).stats['output row-major'],[1,2,3,4,8,12]);
 const invalid=evaluate('broadcasting-semantics',[3,1,0]);assert.equal(invalid.stats['output shape'],null);assert.equal(invalid.stats['output row-major'],null);assert.deepEqual(invalid.curves,[]);
});
test('CP and independent Tucker reconstruction agree for signed, zero and tied weights',()=>{
 for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let keep=0;keep<=2;keep++){
  const result=evaluate('tensor-cp-tucker',[a,b,keep]);close(result.stats['CP–Tucker residual'] as number,0);
  const sorted=[a,b].map(Math.abs).sort((x,y)=>y-x);close(result.stats['approximation error'] as number,Math.hypot(...sorted.slice(keep)));if(keep===2)result.table.rows.forEach(row=>assert.equal(row[3],row[4]));
 }
 assert.equal(evaluate('tensor-cp-tucker',[2,1,2]).stats['Tucker entries'],20);
});
test('TT gauge transformations preserve entries and allocated bonds can exceed minimal ranks',()=>{
 for(const t of [-2,0,1,2])for(const g of [.25,.7,1,1.7,2]){
  const result=evaluate('tensor-trains',[2,t,g]);close(result.stats['reconstruction error'] as number,0);result.table.rows.forEach(row=>close(Number(row[3]),Number(row[4])));assert.equal(result.stats['stored core entries'],16);
  assert.deepEqual(result.stats['actual unfolding ranks'],[1,t===0?1:2,t===0?1:2,1]);
 }
 assert.equal(evaluate('tensor-trains',[1,1,1]).stats['stored core entries'],6);assert.equal(evaluate('tensor-trains',[1,1,1]).stats['reconstruction error'],1);
});
test('every small matrix-chain shape agrees under association and counts actual products',()=>{
 for(let n=1;n<=4;n++)for(let m=1;m<=4;m++)for(let p=1;p<=4;p++)for(let q=1;q<=4;q++){
  const result=evaluate('tensor-networks-contraction',[n,m,p,q]);assert.equal(result.stats['association residual'],0);assert.equal(result.stats['left scalar multiplications'],n*m*p+n*p*q);assert.equal(result.stats['right scalar multiplications'],m*p*q+n*m*q);assert.equal(result.table.rows.length,n*q);
 }
 assert.deepEqual(matrixProduct([1,2,3,4],[5,6],2,2,1).values,[17,39]);
});
test('orthogonal approximation uses squared discarded weights without inventing zero-input energy',()=>{
 const one=evaluate('tensor-low-rank-approximation',[3,1,1]);assert.equal(one.stats['squared error'],1);assert.equal(one.stats['retained energy fraction'],.9);
 assert.equal(evaluate('tensor-low-rank-approximation',[0,0,1]).stats['retained energy fraction'],null);assert.equal(evaluate('tensor-low-rank-approximation',[3,1,0]).stats['retained energy fraction'],0);close(evaluate('tensor-low-rank-approximation',[-3,1,0]).stats['approximation error'] as number,Math.sqrt(10));
});
test('symmetric and skew projections reconstruct arbitrary inputs and are orthogonal',()=>{
 for(let a=-3;a<=3;a++)for(let b=-3;b<=3;b++)for(let c=-3;c<=3;c++){
  const result=evaluate('tensor-symmetry-antisymmetry',[a,b,c,2]);assert.equal(result.stats['reconstruction residual'],0);assert.equal(result.stats['Frobenius inner product'],0);assert.equal(result.stats['vᵀKv for v=(1,2)'],0);result.table.rows.forEach(row=>assert.equal(row[1],Number(row[2])+Number(row[3])));
 }
});
test('storage counts match core allocations, byte modes and expansion counterexample',()=>{
 const result=evaluate('tensor-storage-cost',[6,4,2,1]);assert.equal(result.stats['dense scalars'],4096);assert.equal(result.stats['CP scalars incl. weights'],50);assert.equal(result.stats['Tucker scalars'],112);assert.equal(result.stats['TT core scalars'],80);assert.equal(result.stats['TT bytes'],320);
 assert.equal(evaluate('tensor-storage-cost',[6,4,2,0]).stats['TT bytes'],160);assert.equal(evaluate('tensor-storage-cost',[2,2,4,1]).stats['TT core scalars'],16);
 for(let d=2;d<=8;d++)for(let r=1;r<=4;r++){let count=0;const bonds=[1,...Array(d-1).fill(r),1];for(let k=0;k<d;k++)count+=bonds[k]*8*bonds[k+1];assert.equal(evaluate('tensor-storage-cost',[d,8,r,1]).stats['TT core scalars'],count);}
});
for(const kind of tensorKinds)test(`tensor controls, guided examples and finite geometry: ${kind}`,()=>{
 const spec=tensorLabs[kind];assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);
 for(const preset of spec.presets){const r=evaluate(kind,preset.values);assert.ok(r.curves.flat(2).every(Number.isFinite));r.table.rows.flat().forEach(v=>assert.ok(typeof v==='string'||Number.isFinite(v)));r.network?.nodes.forEach(n=>assert.ok(Number.isFinite(n.x)&&Number.isFinite(n.y)));}
 assert.throws(()=>evaluate(kind,[]));const p=spec.controls.map(c=>c.initial);p[0]=NaN;assert.throws(()=>evaluate(kind,p));
});
test('malformed shape and fractional tensor controls fail rather than silently round',()=>{
 assert.throws(()=>broadcastShape([-1],[1]));assert.throws(()=>broadcastShape([1.5],[]));assert.throws(()=>contractCube(1,[1]));assert.throws(()=>matrixProduct([1],[1],2,1,1));assert.throws(()=>evaluate('tensor-indices-contractions',[.5,1,2]));assert.throws(()=>evaluate('tensor-trains',[2,1,0]));
});
