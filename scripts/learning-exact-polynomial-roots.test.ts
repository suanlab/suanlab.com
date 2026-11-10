import test from 'node:test';
import assert from 'node:assert/strict';
import { exactQ as q,exactAdd as add,exactMul as mul,exactCompare as cmp,exactPolynomialEvaluate as value,isolateExactPolynomialRoots as isolate, type ExactQ } from '../src/lib/learning-exact-polynomial-roots';
const Z=BigInt(0),O=BigInt(1),Q=(n:number,d=1)=>q(BigInt(n),BigInt(d));
function product(a:ExactQ[],b:ExactQ[]){const p=Array.from({length:a.length+b.length-1},()=>Q(0));a.forEach((x,i)=>b.forEach((y,j)=>p[i+j]=add(p[i+j],mul(x,y))));return p;}
const linear=(n:number,d=1)=>[Q(-n,d),Q(1)];
test('exact gcd chains preserve arbitrary multiplicities and zero roots without duplicates',()=>{
 let p=[Q(1)];for(const [root,times] of [[-2,2],[0,3],[1,1],[3,4]])for(let j=0;j<times;j++)p=product(p,linear(root));
 const r=isolate(p);assert.equal(r.degree,10);assert.equal(r.distinctRootCount,4);assert.deepEqual(r.roots.map(x=>x.multiplicity),[2,3,1,4]);
 r.roots.forEach((x,i)=>{const truth=Q([-2,0,1,3][i]);assert.ok(cmp(x.lower,truth)<=Z&&cmp(x.upper,truth)>=Z);});
});
test('irrational roots are complete ordered sign brackets; complex roots are excluded',()=>{
 const p=product([Q(-2),Q(0),Q(1)],[Q(1),Q(0),Q(1)]),r=isolate(p,{precisionBits:64});assert.equal(r.distinctRootCount,2);
 for(const x of r.roots){assert.equal(x.multiplicity,1);assert.ok(value(p,x.lower).n*value(p,x.upper).n<Z);assert.ok(cmp(mul(add(x.upper,{n:-x.lower.n,d:x.lower.d}),q(O<<BigInt(64))),x.lower.n>Z?x.lower:{n:-x.upper.n,d:x.upper.d})<=Z);}
 assert.ok(cmp(r.roots[0].upper,r.roots[1].lower)<Z);
});
test('roots separated by 2^-80 stay distinct and retain multiplicities',()=>{
 const a=Q(1),b=add(a,q(O,O<<BigInt(80)));let p=[Q(1)];for(const root of [a,a,b])p=product(p,[{n:-root.n,d:root.d},Q(1)]);
 const r=isolate(p,{precisionBits:112});assert.equal(r.distinctRootCount,2);assert.deepEqual(r.roots.map(x=>x.multiplicity),[2,1]);
 for(const [i,truth] of [a,b].entries()){assert.ok(cmp(r.roots[i].lower,truth)<=Z);assert.ok(cmp(r.roots[i].upper,truth)>=Z);}
});
test('rational midpoint roots split open Sturm intervals without being counted twice',()=>{
 for(const roots of [[-1,0,1],[-2,-1,0,1,2],[-4,-2,0,2,4]]){let p=[Q(1)];roots.forEach(x=>p=product(p,linear(x)));const r=isolate(p);assert.equal(r.distinctRootCount,roots.length);r.roots.forEach((x,i)=>assert.ok(cmp(x.lower,Q(roots[i]))<=Z&&cmp(x.upper,Q(roots[i]))>=Z));}
});
test('a subnormal-sized rational root can be isolated without merging it into zero',()=>{
 const tiny=q(O,O<<BigInt(1074)),r=isolate(product([Q(0),Q(1)],[{n:-tiny.n,d:tiny.d},Q(1)]),{precisionBits:64});assert.equal(r.distinctRootCount,2);assert.equal(r.roots[0].lower.n,Z);assert.ok(r.roots[1].lower.n>Z);assert.ok(cmp(r.roots[1].lower,tiny)<=Z&&cmp(r.roots[1].upper,tiny)>=Z);
});
test('constant, identically zero and resource-limited preparations are explicit',()=>{
 assert.deepEqual(isolate([Q(2)]).roots,[]);assert.throws(()=>isolate([Q(0)]),RangeError);assert.throws(()=>isolate([Q(-2),Q(0),Q(1)],{maximumBisections:1}),RangeError);assert.throws(()=>isolate([Q(1)],{precisionBits:1}),RangeError);
});
