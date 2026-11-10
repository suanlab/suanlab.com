import test from 'node:test';import assert from 'node:assert/strict';
import {isolateTensorSpectrum as isolate}from'../src/lib/learning-tensor-roots';
import{f,zero,one,add,neg,eq,le,polyMul,sub}from'../src/lib/learning-quantum-hilbert-core';
const polynomial=(roots:ReturnType<typeof f>[])=>roots.reduce((p,r)=>polyMul(p,[neg(r),one]),[one]).reverse();
test('square-free Sturm certificates distinguish multiplicity,zero and non-dyadic rational roots',()=>{
 for(const roots of [[zero],[one,zero,zero,zero],[f(BigInt(1),BigInt(3)),f(BigInt(1),BigInt(3)),f(BigInt(1),BigInt(3))],[f(BigInt(1),BigInt(17)),f(BigInt(2),BigInt(17)),f(BigInt(3),BigInt(17)),f(BigInt(11),BigInt(17))]]){
  const r=isolate(polynomial(roots)),ordered=roots.slice().sort((a,b)=>le(a,b)?-1:1);assert.equal(r.roots.reduce((n,v)=>n+v.multiplicity,0),roots.length);let offset=0;for(const root of r.roots){for(let i=0;i<root.multiplicity;i++)assert(le(root.lower,ordered[offset+i])&&le(ordered[offset+i],root.upper));offset+=root.multiplicity;}
 }
});
test('irrational cubic and quartic spectra retain all real roots instead of using Number approximations',()=>{
 const p=[one,zero,f(BigInt(-5)),zero,f(BigInt(4))];const r=isolate(p);assert.equal(r.roots.length,4);for(const[i,v]of r.roots.entries()){const expected=f(BigInt([-2,-1,1,2][i]));assert(le(v.lower,expected)&&le(expected,v.upper));}
 for(const poly of [[one,zero,f(BigInt(-3)),one],[one,zero,f(BigInt(-5)),zero,one]]){const generic=isolate(poly);assert.equal(generic.roots.length,poly.length-1);assert(generic.roots.every(v=>!v.exact&&v.multiplicity===1));}
 const irrational=isolate([one,zero,f(BigInt(-2))]);assert.equal(irrational.roots.length,2);for(const v of irrational.roots){assert(!v.exact);assert(le(sub(v.upper,v.lower),f(BigInt(1),BigInt(1)<<BigInt(1024))));}
});
test('distinct very close rational roots do not become a repeated eigenspace',()=>{
 const a=f(BigInt(1),BigInt(3)),b=add(a,f(BigInt(1),BigInt(10)**BigInt(180))),c=f(BigInt(2),BigInt(3)),r=isolate(polynomial([a,b,c]));assert.equal(r.roots.length,3);assert(r.roots.every(v=>v.multiplicity===1));assert(le(r.roots[0].upper,r.roots[1].lower));
});
test('root validation rejects nonreal spectra,invalid bounds and unsupported degrees',()=>{
 assert.throws(()=>isolate([one,zero,one]),RangeError);assert.throws(()=>isolate([zero,one]),RangeError);assert.throws(()=>isolate([one]),RangeError);assert.throws(()=>isolate([one,zero,zero,zero,zero,one]),RangeError);assert.throws(()=>isolate([one,zero],511),RangeError);assert.throws(()=>isolate([one,zero],4097),RangeError);assert.throws(()=>isolate([one,{n:BigInt(1),d:BigInt(0)}]),RangeError);
 const repeated=isolate(polynomial([f(BigInt(-2)),f(BigInt(-2)),f(BigInt(1)),f(BigInt(1))]));assert(repeated.roots.every(v=>v.exact&&v.multiplicity===2));assert(eq(repeated.roots[0].lower,f(BigInt(-2))));
});

test('4096-bit isolation uses bounded call depth while retaining close distinct roots and multiplicities',()=>{
 const a=f(BigInt(1),BigInt(3)),b=add(a,f(BigInt(1),BigInt(10)**BigInt(300))),r=isolate(polynomial([neg(one),a,b,b]),4096);
 assert.equal(r.roots.length,3);assert.deepEqual(r.roots.map(v=>v.multiplicity),[1,1,2]);
 for(const v of r.roots)assert(le(sub(v.upper,v.lower),f(BigInt(1),BigInt(1)<<BigInt(4096))));
 assert(le(r.roots[1].upper,r.roots[2].lower));
 const irrational=isolate([one,zero,f(BigInt(-2))],4096);assert.equal(irrational.roots.length,2);assert(irrational.roots.every(v=>!v.exact));
});
