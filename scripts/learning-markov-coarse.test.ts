import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync}from'node:fs';
import {markovCoarseModel as run,isMarkovCoarseParameters as valid,type MarkovCoarseParameters}from'../src/lib/learning-markov-coarse';
import {isMasterObservation}from'../src/lib/learning-master-equation-protocol';
import type{PartitionObservation as Q}from'../src/lib/learning-partition-exact-core';
const refs=JSON.parse(readFileSync('scripts/fixtures/markov-coarse-reference.json','utf8')) as{cases:{label:string;input:MarkovCoarseParameters;expected:unknown}[]};let comparisons=0;
function compare(a:unknown,b:unknown,path='root'){
 if(typeof b==='string'){const q=a as Q;assert(isMasterObservation(q,true),path);assert(q.bounds);const[n,d='1']=b.split('/');for(const end of [q.bounds.lower,q.bounds.upper]){assert.equal(end.numerator.exponent,0);assert.equal(end.denominator.exponent,0);assert.equal(BigInt(end.numerator.coefficient)*BigInt(d),BigInt(n)*BigInt(end.denominator.coefficient),path);}comparisons++;return;}
 if(Array.isArray(b)){assert(Array.isArray(a));assert.equal(a.length,b.length);b.forEach((v,i)=>compare(a[i],v,`${path}.${i}`));return;}
 if(b&&typeof b==='object'){assert(a&&typeof a==='object');for(const[k,v]of Object.entries(b))compare((a as Record<string,unknown>)[k],v,`${path}.${k}`);return;}assert.equal(a,b,path);
}
const find=(label:string)=>refs.cases.find(c=>c.label===label)!.input;
for(const c of refs.cases)test(`exact scalar-hidden reference: ${c.label}`,()=>{
 const r=run(c.input);compare(r,c.expected,c.label);assert.doesNotThrow(()=>JSON.stringify(r));for(const row of r.history){assert(row.micro.every(q=>q.sign===0||q.sign===1));assert.equal(row.microMassResidual.resolution,'structural-zero');assert.equal(row.macroMassResidual.resolution,'structural-zero');assert(row.reconstructionResidual.every(q=>q.resolution==='structural-zero'));}
});
test('same macro law gives distinct next outcomes; matching a stationary law does not establish closure',()=>{
 const a=run(find('stationary-matched nonlumpable')),b=run(find('same macro different hidden')),stationary=run(find('stationary law'));
 assert.deepEqual(a.history[0].macro,b.history[0].macro);assert.equal(a.history[1].macro[1].value,.25);assert.equal(b.history[1].macro[1].value,.75);assert.equal(a.history[1].witnessDistanceTV.value,.5);assert(!a.classification.strongLumpableAllInitialLaws);assert(stationary.history.every(h=>h.naiveErrorTV.resolution==='structural-zero'));assert.equal(a.hiddenEigenvalue.value,0);
});
test('strong lumpability removes observable hidden forcing and memory for every lifting',()=>{
 for(const label of ['strong lumpability','strong other lifting','pure state1 equal exits']){const r=run(find(label));assert(r.classification.strongLumpableAllInitialLaws);for(const h of r.history){assert.equal(h.naiveErrorTV.resolution,'structural-zero');assert(h.initialHiddenTerm.every(q=>q.resolution==='structural-zero'));assert(h.fullMemoryTerm.every(q=>q.resolution==='structural-zero'));assert.equal(h.witnessDistanceTV.resolution,'structural-zero');}}
});
test('full memory reconstructs the exact trajectory, while zero initial hidden state alone is insufficient',()=>{
 const full=run(find('full finite memory'));assert(full.history.every(h=>h.truncatedErrorHalfL1.resolution==='structural-zero'));
 const lifted=run(find('lifted initial law'));assert(lifted.classification.initialHiddenExactlyZero);assert.equal(lifted.history[1].naiveErrorTV.value,0);assert.equal(lifted.history[2].naiveErrorTV.sign,1);
 const a=run(find('stationary-matched nonlumpable')),b=run(find('lift boundary zero'));assert.deepEqual(a.history.map(h=>h.macro),b.history.map(h=>h.macro));assert.notDeepEqual(a.instantaneousMacroTransition,b.instantaneousMacroTransition);
});
test('depth and diagnostic flags do not invent continuous time or random forcing',()=>{
 const zero=run(find('zero steps'));assert.equal(zero.history.length,1);assert.equal(zero.memoryKernels.length,0);assert(!zero.classification.continuousTimeModel);assert(!zero.classification.physicalNoiseSpecified);assert(!zero.classification.kernelsAreProbabilities);
 const a=run(find('slow alternating hidden'));assert(a.classification.hiddenModeAlternates);assert(a.classification.geometricHiddenModeStrictlyDecays);assert.equal(run(find('persistent hidden mode')).hiddenEigenvalue.value,1);
});
test('preparations reject invalid probabilities, sparse weights, nonintegers and foreign fields',()=>{
 const p=find('stationary-matched nonlumpable'),before=structuredClone(p),r=run(p);assert.deepEqual(p,before);assert(Object.isFrozen(r.parameters.initialWeights));
 for(const patch of [{weight01:5},{weight01:-1},{transitionDenominator:0},{liftNumerator:3},{steps:33},{memoryDepth:-1},{initialWeights:[0,0,0]},{initialWeights:[1,0]},{initialWeights:[NaN,0,0]},{steps:.5}]){assert(!valid({...p,...patch}));assert.throws(()=>run({...p,...patch}));}
 const sparse=Array(3);sparse[0]=1;assert(!valid({...p,initialWeights:sparse}));assert(!valid({...p,extra:1}));assert(!valid({...p,[Symbol('extra')]:1}));
});
test('reference inventory records exact rational equality, separate from rounded observations',()=>{assert.equal(refs.cases.length,30);mkdirSync('.runtime/learning/coarse-graining-foundation',{recursive:true});writeFileSync('.runtime/learning/coarse-graining-foundation/reference-checks.json',JSON.stringify({preparations:refs.cases.length,records:refs.cases.reduce((n,c)=>n+run(c.input).history.length,0),exactRationalComparisons:comparisons},null,2)+'\n');console.log(JSON.stringify({preparations:refs.cases.length,exactRationalComparisons:comparisons}));});

test('truncation may leave probability space; signed values are retained instead of clipped',()=>{
 const r=run(find('negative truncated pseudo-law'));assert.equal(r.history[2].initialAwareTruncated[0].value,1.25);assert.equal(r.history[2].initialAwareTruncated[1].value,-.25);assert(!r.history[2].truncatedIsProbability);assert.equal(r.history[2].macro[0].value,1);assert.equal(r.history[2].truncatedErrorHalfL1.value,.25);
});
test('a hidden memory coefficient may be exactly nonzero below ordinary Number range',()=>{
 const r=run(find('nonzero underflow memory')),q=r.memoryKernels[31].matrix[0][0];assert.equal(r.hiddenEigenvalue.value,1e-12);assert.equal(q.sign,1);assert(q.value===null||q.value===0);assert(q.bounds);assert.notEqual(q.bounds.lower.numerator.coefficient,'0');
});
