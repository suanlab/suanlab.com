import test from 'node:test';
import assert from 'node:assert/strict';
import {createRitzIntervalContext}from'../src/lib/learning-ritz-interval';
import type{PartitionObservation as Q}from'../src/lib/learning-partition-exact-core';
type F={n:bigint;d:bigint};const f=(n:number,d=1):F=>({n:BigInt(n),d:BigInt(d)}),le=(a:F,b:F)=>a.n*b.d<=b.n*a.d;
function bounds(q:Q){assert(q.bounds);const read=(v:NonNullable<Q['bounds']>['lower']):F=>{const e=v.numerator.exponent-v.denominator.exponent,n=BigInt(v.numerator.coefficient),d=BigInt(v.denominator.coefficient);return e>=0?{n:n*BigInt(10)**BigInt(e),d}:{n,d:d*BigInt(10)**BigInt(-e)};};return{lower:read(q.bounds.lower),upper:read(q.bounds.upper)};}
function contains(q:Q,value:F){const r=bounds(q);assert(le(r.lower,value)&&le(value,r.upper),JSON.stringify(q));}
test('Ritz fixed-grid arithmetic encloses exact signed rational operations without Number conversion',()=>{
 const c=createRitzIntervalContext();let checked=0;
 for(let a=-17;a<=17;a++)for(let d=1;d<=11;d++)for(const b of [-19,-1,1,23]){
  const x=f(a,d),y=f(b,13),ix=c.from(x),iy=c.from(y);contains(c.observation(ix),x);contains(c.observation(c.add(ix,iy)),{n:x.n*y.d+y.n*x.d,d:x.d*y.d});contains(c.observation(c.sub(ix,iy)),{n:x.n*y.d-y.n*x.d,d:x.d*y.d});contains(c.observation(c.multiply(ix,iy)),{n:x.n*y.n,d:x.d*y.d});const n=x.n*y.d,den=x.d*y.n;contains(c.observation(c.divide(ix,iy)),den<0?{n:-n,d:-den}:{n,d:den});contains(c.observation(c.square(ix)),{n:x.n*x.n,d:x.d*x.d});checked+=6;
 }
 assert.equal(checked,9240);
});
test('complex matrix operations retain conjugation and exact identities in their outward ranges',()=>{
 const c=createRitzIntervalContext(),z=c.complex({re:f(2,3),im:f(0)}),w=c.complex({re:f(-5,11),im:f(7,13)}),A=[[z,w],[c.conjugate(w),c.complex({re:f(3),im:f(0)})]],diff=c.matrixSubtract(A,c.adjoint(A));
 for(const row of diff)for(const v of row){contains(c.observeComplex(v).real,f(0));contains(c.observeComplex(v).imaginary,f(0));}
 const square=c.product(A,A),n=c.normSquared([z,w]);contains(c.observation(c.sub(square[0][0].real,n)),f(0));contains(c.observation(square[0][0].imaginary),f(0));
 const zero=c.matrixSubtract(c.product(A,c.identity(2)),A);for(const row of zero)for(const v of row){contains(c.observeComplex(v).real,f(0));contains(c.observeComplex(v).imaginary,f(0));}
});
test('probability uses only its proven mathematical range and disjoint enclosures fail explicitly',()=>{
 const c=createRitzIntervalContext();contains(c.probability(c.range({lower:f(-1,7),upper:f(2,3)})),f(0));contains(c.probability(c.range({lower:f(2,3),upper:f(9,7)})),f(1));assert.throws(()=>c.probability(c.from(f(2))),RangeError);assert.throws(()=>c.probability(c.from(f(-1))),RangeError);
});
test('invalid rational inputs,unordered endpoints and zero-crossing divisors fail before arithmetic',()=>{
 const c=createRitzIntervalContext();assert.throws(()=>c.from(f(1,0)),RangeError);assert.throws(()=>c.from(f(1,-1)),RangeError);assert.throws(()=>c.range({lower:f(2),upper:f(1)}),RangeError);assert.throws(()=>c.divide(c.one,{lower:-BigInt(1),upper:BigInt(1)}),RangeError);assert.throws(()=>c.observation({lower:BigInt(2),upper:BigInt(1)}),RangeError);
});
