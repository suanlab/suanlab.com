import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync}from'node:fs';
import reference from './fixtures/quantum-information-capacity-independent.json';
import {qubitDephasingCapacity}from'../src/lib/learning-quantum-information-capacity';
import {f,zero,one,add,sub,le,q,type F}from'../src/lib/learning-quantum-hilbert-core';
import type{PartitionObservation,PartitionRational}from'../src/lib/learning-partition-exact-core';
import {getLearningContent}from'../src/lib/learning-content';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const bloch=(r:string[])=>r.map(parse) as [F,F,F];
function decimal(s:string):F{
 const[mantissa,exponent='0']=s.toLowerCase().split('e'),negative=mantissa.startsWith('-');
 const[a,b='']=mantissa.replace(/^[-+]/,'').split('.'),n=BigInt(a+b)*(negative?BigInt(-1):BigInt(1)),shift=Number(exponent)-b.length;
 return shift>=0?f(n*BigInt(10)**BigInt(shift)):f(n,BigInt(10)**BigInt(-shift));
}
function decode(x:PartitionRational):F{
 const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);
 return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));
}
function compare(observation:PartitionObservation,expected:string){
 assert(observation.bounds);const center=decimal(expected),allowance=decimal(reference.comparisonAllowance),lower=decode(observation.bounds.lower),upper=decode(observation.bounds.upper);
 assert(le(lower,add(center,allowance)),`lower exceeds ${expected}`);assert(le(sub(center,allowance),upper),`upper below ${expected}`);
 assert(le(sub(upper,lower),decimal('1e-65')),'overly broad interval');
}
for(const item of reference.cases)test(`independent Stinespring/Choi eigensystem: ${item.name}`,()=>{
 const result=qubitDephasingCapacity(parse(item.p),bloch(item.bloch));
 for(const key of Object.keys(item.expected)as(keyof typeof item.expected)[])compare(result[key],item.expected[key]);
 compare(result.rawSelectedInputCoherentInformation,item.expected.selectedInputCoherentInformation);
 compare(result.capacitiesByFamilyTheorem.quantumQubitsPerUse,item.quantumRate);
 compare(result.capacitiesByFamilyTheorem.entanglementAssistedClassicalBitsPerUse,item.assistedRate);
 for(const key of ['outputDensity','environmentDensity','normalizedChoi']as const){
  for(let i=0;i<item[key].length;i++)for(let j=0;j<item[key][i].length;j++){
   compare(result[key][i][j].real,item[key][i][j].real);compare(result[key][i][j].imaginary,item[key][i][j].imaginary);
  }
 }
 assert.deepEqual(result.capacitiesByFamilyTheorem.classicalBitsPerUse,q(one));
 assert.doesNotThrow(()=>JSON.stringify(result));
});
test('identity, complete dephasing and deterministic Z have different endpoint behavior',()=>{
 for(const p of ['0','1']){
  const result=qubitDephasingCapacity(parse(p),bloch(['0','0','0']));
  assert.equal(result.capacitiesByFamilyTheorem.quantumQubitsPerUse.value,1);
  assert.equal(result.capacitiesByFamilyTheorem.entanglementAssistedClassicalBitsPerUse.value,2);
 }
 const full=qubitDephasingCapacity(parse('1/2'),bloch(['1/5','2/5','1/5']));
 assert.equal(full.capacitiesByFamilyTheorem.quantumQubitsPerUse.value,0);
 assert.equal(full.selectedInputCoherentInformation.value,0);assert.equal(full.quantumCapacityStrictlyPositive,false);
 assert.equal(full.zBasisMessageInformation.value,1);
});
test('pure-input coherent information is not a family quantum capacity',()=>{
 const result=qubitDephasingCapacity(parse('1/10'),bloch(['24/25','0','-7/25']));
 assert.equal(result.selectedInputCoherentInformation.value,0);
 assert.equal(result.selectedInputReferenceMutualInformation.value,0);
 assert(result.capacitiesByFamilyTheorem.quantumQubitsPerUse.value!>0.53);
 assert.equal(result.xBasisEqualPriorGuessSuccess.value,0.9);
});
test('small positive capacity is retained when the displayed phase entropy rounds to one',()=>{
 const D=BigInt(10)**BigInt(30),result=qubitDephasingCapacity(f(D+BigInt(1),BigInt(2)*D),bloch(['0','0','0']));
 assert.equal(result.phaseErrorEntropy.value,1);
 assert.equal(result.quantumCapacityStrictlyPositive,true);
 assert.equal(result.capacitiesByFamilyTheorem.quantumQubitsPerUse.sign,1);
 assert(result.capacitiesByFamilyTheorem.quantumQubitsPerUse.value!>0);
});
test('capacity is input-independent and symmetric under a known output Z',()=>{
 const r=qubitDephasingCapacity(parse('1/10'),bloch(['0','0','0']));
 const flipped=qubitDephasingCapacity(parse('9/10'),bloch(['0','0','1']));
 assert.deepEqual(r.capacitiesByFamilyTheorem,flipped.capacitiesByFamilyTheorem);
 assert.notDeepEqual(r.selectedInputCoherentInformation,flipped.selectedInputCoherentInformation);
});
test('internal exact output coefficients may exceed the external raw-input digit budget',()=>{
 const D=BigInt(10)**BigInt(31);
 const r=qubitDephasingCapacity(f(BigInt(1),D+BigInt(1)),[f(BigInt(1),D-BigInt(1)),zero,zero]);
 assert(r.outputDensity);assert(r.environmentEntropy.bounds);
});
test('invalid raw probability and state are rejected',()=>{
 for(const p of ['-1/2','3/2'])assert.throws(()=>qubitDephasingCapacity(parse(p),[zero,zero,zero]),RangeError);
 assert.throws(()=>qubitDephasingCapacity({n:BigInt(1),d:BigInt(0)},[zero,zero,zero]),RangeError);
 assert.throws(()=>qubitDephasingCapacity(parse('1/2'),[one,one,zero]),RangeError);
});
test('capacity manuscript renders in both languages without broken tables or mathematics',async()=>{
 const id='quantum-channel-capacities-coding-limits',content=await getLearningContent(id);
 for(const locale of ['ko','en']as const){assert.equal(content.outline[locale].length,12);assert(!content[locale].includes('katex-error'));assert(content[locale].includes('<table'));}
 assert(!/[가-힣]/.test(readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`,'utf8')));
});

import jointReference from './fixtures/quantum-joint-entropy-independent.json';
import {correlatedBellEntropy}from'../src/lib/learning-quantum-information-joint-entropy';
for(const item of jointReference.cases)test(`independent joint entropy/PPT: c=${item.c},t=${item.t}`,()=>{
 const r=correlatedBellEntropy(parse(item.c),parse(item.t));compare(r.jointEntropy,item.jointEntropy);
 assert.equal(r.entangled,item.entangled);assert.deepEqual(r.negativity,q(parse(item.negativity)));
 for(let i=0;i<4;i++)for(let j=0;j<4;j++)assert.deepEqual(r.density[i][j],{real:q(parse(item.density[i][j])),imaginary:q(zero)});
 const order=(a:F,b:F)=>a.n*b.d<b.n*a.d?-1:a.n*b.d>b.n*a.d?1:0;
 assert.deepEqual(r.partialTransposeEigenvalues.map(v=>decode(v.bounds!.lower)).sort(order),item.partialTransposeEigenvalues.map(parse).sort(order));
});
test('canonical Bell, correlated and independent preparations have fixed local entropy',()=>{
 for(const[c,t,S,conditional,mutual]of [['1','1',0,-1,2],['1','0',1,0,1],['0','1',2,1,0]]as const){
  const r=correlatedBellEntropy(parse(c),parse(t));
  assert.equal(r.localEntropyA.value,1);assert.equal(r.localEntropyB.value,1);
  assert.equal(r.jointEntropy.value,S);assert.equal(r.conditionalEntropy.value,conditional);assert.equal(r.mutualInformation.value,mutual);
 }
});
test('positive conditional entropy does not imply separability',()=>{
 const r=correlatedBellEntropy(parse('1/2'),one);
 assert(r.conditionalEntropy.value!>0.54);assert(r.entangled);assert.deepEqual(r.negativity,q(parse('1/8')));
});
test('joint-family invalid parameters fail and incoherent entropy gain is exactly zero',()=>{
 assert.throws(()=>correlatedBellEntropy(parse('3/2'),one),RangeError);
 assert.throws(()=>correlatedBellEntropy(one,parse('-1/2')),RangeError);
 assert.equal(correlatedBellEntropy(parse('4/5'),zero).dephasingEntropyGain.value,0);
});
