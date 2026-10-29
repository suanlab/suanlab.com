import test from 'node:test';
import assert from 'node:assert/strict';
import { abstractAlgebraKinds, evaluateAbstractAlgebra as evaluate, residueInverse, binaryPolynomialProduct as mul, squareAction } from '../src/lib/learning-abstract-algebra';
import { abstractAlgebraLabs, learningGuides } from '../src/data/interactive-book';
const close=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('units match a full multiplication permutation, including zero divisors and zero',()=>{
 for(let n=2;n<=9;n++)for(let a=0;a<n;a++){
  const outputs=new Set(Array.from({length:n},(_,x)=>a*x%n)),inverse=residueInverse(a,n);
  assert.equal(inverse!==null,outputs.size===n);if(inverse!==null)assert.equal(a*inverse%n,1);
 }
 assert.equal(residueInverse(2,6),null);assert.equal(residueInverse(0,5),null);assert.equal(residueInverse(2,5),3);
});
test('residue homomorphism condition is equivalent to representative-independent addition on every pair',()=>{
 for(let m=2;m<=8;m++)for(let n=2;n<=8;n++)for(let k=0;k<=7;k++){
  let preserves=true;for(let x=0;x<m;x++)for(let y=0;y<m;y++)if((k*((x+y)%m))%n!==(k*x+k*y)%n)preserves=false;
  const r=evaluate('homomorphisms-isomorphisms',[m,n,k]);assert.equal(r.stats['1{homomorphism}'],Number(preserves));
  if(preserves)assert.equal((r.stats.kernel as number[]).length*(r.stats.image as number[]).length,m);else {assert.equal(r.stats.kernel,null);assert.equal(r.stats['1{isomorphism}'],null);}
 }
});
test('cosets form a disjoint cover and quotient labels respect all additions',()=>{
 for(let n=2;n<=9;n++)for(let h=0;h<n;h++){
  const r=evaluate('subgroups-quotients',[n,h]),H=r.stats['subgroup H'] as number[],g=r.stats['coset count'] as number;assert.equal(H.length*g,n);
  const seen=new Set<string>();r.table.rows.forEach(row=>seen.add(String(row[2])));assert.equal(seen.size,g);
  for(let x=0;x<n;x++)for(let y=0;y<n;y++)assert.equal(((x+y)%n)%g,((x%g)+(y%g))%g);
 }
});
test('binary polynomial products obey ring laws and distinguish an irreducible cubic from a zero-divisor quotient',()=>{
 for(const m of [10,11])for(let a=0;a<8;a++)for(let b=0;b<8;b++)for(let c=0;c<8;c++){
  assert.equal(mul(a,mul(b,c,m),m),mul(mul(a,b,m),c,m));assert.equal(mul(a,b^c,m),mul(a,b,m)^mul(a,c,m));assert.equal(mul(a,b,m),mul(b,a,m));
 }
 assert.equal(mul(2,4,11),3);assert.equal(mul(2,5,11),1);assert.equal(mul(2,5,10),0);
 for(let a=1;a<8;a++)assert.equal(new Set(Array.from({length:8},(_,b)=>mul(a,b,11))).size,8);
});
test('module kernels match actual distinct images and field dimensions do not describe a composite scalar ring',()=>{
 for(let n=2;n<=7;n++)for(let a=0;a<=6;a++)for(let b=0;b<=6;b++){
  const r=evaluate('modules-linear-actions',[n,a,b]),images=new Set(r.table.rows.map(row=>`${row[2]},${row[3]}`));assert.equal(images.size,r.stats['image size']);assert.equal(Number(r.stats['kernel size'])*images.size,n*n);
 }
 assert.equal(evaluate('modules-linear-actions',[6,2,3]).stats['field dimension of kernel'],null);assert.equal(evaluate('modules-linear-actions',[5,0,0]).stats['field dimension of kernel'],2);
});
test('square actions obey D4 composition and orbit stabilizer for all selectable points',()=>{
 for(let r=0;r<4;r++)for(let f=0;f<2;f++)for(let s=0;s<4;s++)for(let h=0;h<2;h++){
  const v=[1,2],left=squareAction(r,f,squareAction(s,h,v)),rr=((r+(f?-s:s))%4+4)%4;assert.deepEqual(left,squareAction(rr,(f+h)%2,v));
 }
 assert.notDeepEqual(squareAction(1,0,squareAction(0,1,[1,0])),squareAction(0,1,squareAction(1,0,[1,0])));
 for(let point=0;point<3;point++){const r=evaluate('algebraic-symmetries',[1,1,point]);assert.equal(r.stats['orbit × stabilizer'],8);r.network!.edges.forEach(e=>assert.ok(r.network!.nodes.some(n=>n.id===e.to)));}
});
test('finite-field system solutions satisfy both constraints and agree with independent root substitution',()=>{
 for(let mode=0;mode<2;mode++){const n=mode===0?5:7;for(let s=0;s<n;s++)for(let q=0;q<n;q++){
  const roots=Array.from({length:n},(_,x)=>x).filter(x=>((x*x-s*x+q)%n+n)%n===0),r=evaluate('polynomial-systems',[mode,s,q]);assert.equal(r.stats['solution count'],roots.length);const expected=roots.flatMap(x=>[x,((s-x)%n+n)%n]);assert.deepEqual(r.stats['solutions flattened'],expected);
 }}
 assert.equal(evaluate('polynomial-systems',[0,0,2]).stats['solution count'],0);
});
test('group-averaged features are invariant while linear averages collapse distinct inputs',()=>{
 for(let x=-2;x<=2;x++)for(let y=-2;y<=2;y++)for(let f=0;f<3;f++){
  const r=evaluate('invariant-constructions',[x,y,f]),rot=evaluate('invariant-constructions',[-y,x,f]);assert.equal(r.stats['averaged invariant'],rot.stats['averaged invariant']);assert.equal(r.stats['averaged invariant'],f===1?(x*x+y*y)/2:0);
 }
});
test('conjugation projection agrees with the commutant formula without granting reflection equivariance',()=>{
 for(let a=-2;a<=2;a++)for(let b=-2;b<=2;b++)for(let c=-2;c<=2;c++)for(let d=-2;d<=2;d++){
  const r=evaluate('algebraic-representations-scope',[a,b,c,d]);assert.deepEqual(r.stats['C₄ projected row-major'],[(a+d)/2,(b-c)/2,(c-b)/2,(a+d)/2]);close(r.stats['projected C₄ residual'] as number,0);
 }
 assert.ok(Number(evaluate('algebraic-representations-scope',[1,-1,1,1]).stats['projected reflection residual'])>0);assert.equal(evaluate('algebraic-representations-scope',[1,0,0,1]).stats['projected reflection residual'],0);
});
for(const kind of abstractAlgebraKinds)test(`abstract algebra presets and input domains: ${kind}`,()=>{
 assert.equal(learningGuides[kind].steps.length,4);assert.equal(learningGuides[kind].experiments.length,3);const spec=abstractAlgebraLabs[kind];for(const p of spec.presets){const r=evaluate(kind,p.values);assert.ok(r.curves.flat(2).every(Number.isFinite));assert.ok(r.table.rows.length>0);}assert.throws(()=>evaluate(kind,[]));const params=spec.controls.map(c=>c.initial);params[0]=NaN;assert.throws(()=>evaluate(kind,params));params[0]=spec.controls[0].initial+.5;assert.throws(()=>evaluate(kind,params));
});
