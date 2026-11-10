import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import reference from './fixtures/quantum-information-entropy-independent.json';
import { binaryQubitEnsemble } from '../src/lib/learning-quantum-information-ensemble';
import { qubitRelativeEntropy, qubitDephasingDataProcessing } from '../src/lib/learning-quantum-information-relative-entropy';
import { binaryEntropyRange, point, log2Range } from '../src/lib/learning-quantum-information-entropies';
import { f, zero, one, add, sub, le, type F } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { getLearningContent } from '../src/lib/learning-content';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const bloch=(s:string[])=>s.map(parse) as [F,F,F];
function decimal(s:string):F {
 const [mantissa,exponent='0']=s.toLowerCase().split('e'),negative=mantissa.startsWith('-');
 const [whole,fraction='']=mantissa.replace(/^[-+]/,'').split('.');
 const digits=BigInt(whole+fraction)*(negative?BigInt(-1):BigInt(1)),shift=Number(exponent)-fraction.length;
 return shift>=0?f(digits*BigInt(10)**BigInt(shift)):f(digits,BigInt(10)**BigInt(-shift));
}
function decode(x:PartitionRational):F {
 const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);
 return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));
}
function compare(observation:PartitionObservation,expected:string) {
 assert(observation.bounds);
 const lower=decode(observation.bounds.lower),upper=decode(observation.bounds.upper),center=decimal(expected),allowance=decimal(reference.comparisonAllowance);
 assert(le(lower,add(center,allowance)),`lower > reference ${expected}`);
 assert(le(sub(center,allowance),upper),`upper < reference ${expected}`);
 assert(le(sub(upper,lower),decimal('1e-65')),'unreasonably broad enclosure');
}
for(const item of reference.ensembles)test(`independent cq matrix reference: ${item.name}`,()=>{
 const result=binaryQubitEnsemble(bloch(item.bloch0),bloch(item.bloch1),parse(item.prior),item.measurement as 'helstrom'|'z');
 for(const key of Object.keys(item.expected) as (keyof typeof item.expected)[])compare(result[key],item.expected[key]);
 compare(result.rawMeasurementInformation,item.expected.selectedMeasurementInformation);
 compare(result.rawHolevoInformation,item.expected.holevoInformation);
 assert.doesNotThrow(()=>JSON.stringify(result));
});
for(const item of reference.relative)test(`independent matrix logarithm: ${item.name}`,()=>{
 const result=qubitRelativeEntropy(bloch(item.rho),bloch(item.sigma));
 assert.equal(result.status,item.status);
 if(item.status==='finite'){assert(result.value);compare(result.value,item.value!);}
 else assert.equal(result.value,null);
});
test('worked ensemble separates guessing probability, measured information and Holevo information',()=>{
 const result=binaryQubitEnsemble(bloch(['0','0','1']),bloch(['24/25','0','-7/25']),parse('1/2'));
 assert(Math.abs(result.holevoInformation.value!-0.7219280948873623)<1e-15);
 assert(Math.abs(result.selectedMeasurementInformation.value!-0.5310044064107188)<1e-15);
 assert(result.holevoGap.value!>0.19);
});
test('a guessing-optimal constant measurement can lose positive classical information',()=>{
 const a=bloch(['0','0','1/5']),b=bloch(['0','0','-1/5']),p=parse('9/10');
 const guess=binaryQubitEnsemble(a,b,p,'helstrom'),z=binaryQubitEnsemble(a,b,p,'z');
 assert.equal(guess.selectedMeasurementInformation.value,0);
 assert(z.selectedMeasurementInformation.value!>0);
});
test('nonorthogonal pure reference is infinite, while dephased divergence is one bit',()=>{
 const result=qubitDephasingDataProcessing(bloch(['0','0','1']),bloch(['1','0','0']));
 assert.equal(result.before.status,'infinite');assert.equal(result.after.status,'finite');
 assert.equal(result.after.value!.value,1);
});
test('noncommuting worked pair loses log2(5/3) under dephasing',()=>{
 const result=qubitDephasingDataProcessing(bloch(['0','0','3/5']),bloch(['4/5','0','0']));
 assert.equal(result.before.status,'finite');assert.equal(result.after.status,'finite');
 assert(Math.abs(result.before.value!.value!-1.0150374992788438)<1e-15);
 assert(Math.abs(result.before.value!.value!-result.after.value!.value!-Math.log2(5/3))<1e-15);
});
test('a full-rank state arbitrarily close to pure remains finite',()=>{
 const D=BigInt(10)**BigInt(30),result=qubitRelativeEntropy([zero,zero,f(BigInt(-1))],[zero,zero,f(D-BigInt(1),D)]);
 assert.equal(result.status,'finite');assert(result.value!.value!>100);
});
test('entropy interval handles a maximum inside its interval and logarithm domains',()=>{
 const h=binaryEntropyRange({lower:parse('2/5'),upper:parse('3/5')});
 assert.equal(h.upper.n,h.upper.d);assert(le(zero,h.lower));
 assert.throws(()=>binaryEntropyRange({lower:parse('-1/5'),upper:one}),RangeError);
 assert.throws(()=>log2Range(point(zero)),RangeError);
});
for(const id of ['classical-quantum-ensembles-holevo-information','quantum-relative-entropy-data-processing'])test(`bilingual manuscript: ${id}`,async()=>{
 const content=await getLearningContent(id);
 for(const locale of ['ko','en'] as const){assert(content.outline[locale].length>=10);assert(!content[locale].includes('katex-error'));}
 assert(!/[가-힣]/.test(readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`,'utf8')));
});

test('state/prior validation occurs before any entropy clipping',()=>{
 const valid=bloch(['0','0','0']),invalid=bloch(['1','1','0']);
 assert.throws(()=>binaryQubitEnsemble(invalid,valid,parse('1/2')),RangeError);
 assert.throws(()=>binaryQubitEnsemble(valid,valid,parse('3/2')),RangeError);
 assert.throws(()=>qubitRelativeEntropy(valid,invalid),RangeError);
});
test('certain labels give zero ensemble information for either selected measurement',()=>{
 for(const prior of ['0','1'])for(const measurement of ['helstrom','z'] as const){
  const result=binaryQubitEnsemble(bloch(['0','0','1']),bloch(['1','0','0']),parse(prior),measurement);
  assert.equal(result.holevoInformation.value,0);
  assert.equal(result.selectedMeasurementInformation.value,0);
 }
});
