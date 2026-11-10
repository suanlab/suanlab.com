import test from'node:test';import assert from'node:assert/strict';
import{f,q,observe,zero,one}from'../src/lib/learning-quantum-hilbert-core';
import{createQuantumVariationalWireChecks}from'../src/lib/learning-quantum-variational-wire-schema';
import{certifiedPrimarySpectrum,certifiedMinimum}from'../src/lib/learning-quantum-variational-polynomial';
test('wire observations preserve legitimate broad unresolved intervals instead of imposing fixture precision on every input',()=>{
 const c=createQuantumVariationalWireChecks();assert(c.q(observe({lower:f(BigInt(-1)),upper:one})));assert(c.probability(observe({lower:zero,upper:one})));assert(!c.exact(observe({lower:zero,upper:one})));assert(!c.probability(observe({lower:f(BigInt(-1)),upper:one})));
});
test('wire observations bind exact fractions to their numeric projections and logarithmic display',()=>{
 const c=createQuantumVariationalWireChecks(),x=q(f(BigInt(1),BigInt(3)));assert(c.exact(x));assert(!c.q({...x,value:.5}));assert(!c.q({...x,logMagnitude:1000}));const y=structuredClone(x);y.bounds!.lower.denominator.coefficient='-3';assert(!c.q(y));assert(!c.q({...x,[Symbol('foreign')]:0}));
});
test('wire data accessors are rejected without evaluating them',()=>{
 const c=createQuantumVariationalWireChecks(),x=structuredClone(q(one));let reads=0;Object.defineProperty(x,'value',{enumerable:true,get(){reads++;return 1;}});assert(!c.q(x));assert.equal(reads,0);
});
test('exact degree certificates distinguish a polynomial root from its minimum and retain multiplicities',()=>{
 const point=(x:number,multiplicity=1)=>({energy:q(f(BigInt(x))),lower:q(f(BigInt(x))),upper:q(f(BigInt(x))),multiplicity,exact:true}),poly=[one,f(BigInt(-3)),f(BigInt(2))];assert(certifiedPrimarySpectrum([point(1),point(2)],poly));assert(certifiedMinimum(point(1),poly));assert(!certifiedMinimum(point(2),poly));assert(!certifiedPrimarySpectrum([point(1),point(1)],poly));assert(certifiedMinimum(point(1,2),[one,f(BigInt(-2)),one]));assert(!certifiedMinimum(point(1,1),[one,f(BigInt(-2)),one]));
});
