import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import reference from './fixtures/quantum-information-resources-independent.json';
import {repetitionCodeExperiment,repetitionKnillLaflamme} from '../src/lib/learning-quantum-information-error-correction';
import {pureStateMajorization} from '../src/lib/learning-quantum-information-majorization';
import {bellPairDilution} from '../src/lib/learning-quantum-information-bell-dilution';
import {f,q,add,sub,le,type F} from '../src/lib/learning-quantum-hilbert-core';
import {quadratic,circuitScalar as complex,circuitOne as O,circuitZero as Z} from '../src/lib/learning-qubit-scalar';
import {getLearningContent} from '../src/lib/learning-content';
import type{PartitionObservation,PartitionRational}from'../src/lib/learning-partition-exact-core';
const parse=(s:string):F=>{const[n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const input=(list:{real:string[];imaginary:string[]}[])=>list.map(z=>complex(quadratic(parse(z.real[0]),parse(z.real[1])),quadratic(parse(z.imaginary[0]),parse(z.imaginary[1]))));
for(const item of reference.cases)test(`independent dense syndrome branches: ${item.name}`,()=>{
 const result=repetitionCodeExperiment(input(item.input),item.error);
 assert.deepEqual(result.encodedDensity,item.encodedDensity);
 assert.deepEqual(result.erroredDensity,item.erroredDensity);
 assert.deepEqual(result.branches,item.branches);
 assert.doesNotThrow(()=>JSON.stringify(result));
});
for(const item of reference.noise)test(`independent dense noise/reference channel: ${item.name}`,()=>{
 const result=repetitionCodeExperiment(input(item.input),'III',parse(item.p));
 for(const key of ['logicalOutput','normalizedLogicalChoi','inputSquaredFidelity','entanglementFidelity']as const)assert.deepEqual(result[key],item[key]);
});
for(const item of reference.kl)test(`independent KL compressed operators: ${item.words.join(',')}`,()=>{
 const result=repetitionKnillLaflamme(item.words);
 assert.equal(result.conditionHolds,item.conditionHolds);
 for(const[index,pair]of result.pairs.entries())for(const key of ['a','b','compressed','scalarOnCode','c']as const)assert.deepEqual(pair[key],item.pairs[index][key]);
});
function decimal(s:string):F{
 const[mantissa,exponent='0']=s.toLowerCase().split('e'),negative=mantissa.startsWith('-');
 const[a,b='']=mantissa.replace(/^[-+]/,'').split('.'),n=BigInt(a+b)*(negative?BigInt(-1):BigInt(1)),shift=Number(exponent)-b.length;
 return shift>=0?f(n*BigInt(10)**BigInt(shift)):f(n,BigInt(10)**BigInt(-shift));
}
function decode(x:PartitionRational):F{
 const shift=x.numerator.exponent-x.denominator.exponent,n=BigInt(x.numerator.coefficient),d=BigInt(x.denominator.coefficient);
 return shift>=0?f(n<<BigInt(shift),d):f(n,d<<BigInt(-shift));
}
function entropyComparison(observation:PartitionObservation,expected:string){
 assert(observation.bounds);const center=decimal(expected),allowance=decimal('1e-110');
 assert(le(decode(observation.bounds.lower),add(center,allowance)));
 assert(le(sub(center,allowance),decode(observation.bounds.upper)));
 assert(le(sub(decode(observation.bounds.upper),decode(observation.bounds.lower)),decimal('1e-65')));
}
for(const[index,item]of reference.majorization.entries())test(`independent permutation convex hull: ${index}`,()=>{
 const result=pureStateMajorization(item.source.map(parse),item.target.map(parse));
 assert.equal(result.forward.deterministic,item.forward);assert.equal(result.reverse.deterministic,item.reverse);
 assert.deepEqual(result.forward.optimalStochasticSuccess,q(parse(item.pForward)));
 assert.deepEqual(result.reverse.optimalStochasticSuccess,q(parse(item.pReverse)));
 entropyComparison(result.sourceEntropy,item.entropySource);entropyComparison(result.targetEntropy,item.entropyTarget);
});
test('the incomparable pair has opposite violated prefixes and nonreciprocal rates',()=>{
 const r=pureStateMajorization(['1/2','1/4','1/4'].map(parse),['2/5','2/5','1/5'].map(parse));
 assert.equal(r.relation,'incomparable');assert.equal(r.forward.firstViolation,1);assert.equal(r.reverse.firstViolation,2);
 assert.deepEqual(r.forward.optimalStochasticSuccess,q(parse('5/6')));assert.deepEqual(r.reverse.optimalStochasticSuccess,q(parse('4/5')));
 assert(r.targetEntropy.value!>r.sourceEntropy.value!);
});
test('zero target tails, unequal lengths and unsorted spectra are handled explicitly',()=>{
 const r=pureStateMajorization(['0','1/2','1/2'].map(parse),[parse('1')]);
 assert.equal(r.relation,'forward-only');assert.deepEqual(r.sourceSortingIndices,[1,2,0]);
 assert.equal(r.forward.tails[1].applicable,false);assert.equal(r.forward.tails[1].ratio,null);
 assert.deepEqual(r.reverse.optimalStochasticSuccess,q(parse('0')));
});
test('a tiny exact prefix violation survives rounded displayed values',()=>{
 const D=BigInt(10)**BigInt(30),r=pureStateMajorization([f(D+BigInt(1),BigInt(2)*D),f(D-BigInt(1),BigInt(2)*D)],[parse('1/2'),parse('1/2')]);
 assert.equal(r.forward.prefixes[0].source.value,r.forward.prefixes[0].target.value);
 assert.equal(r.forward.deterministic,false);assert.equal(r.forward.firstViolation,1);
});
test('input-specific plus fidelity differs from entanglement fidelity',()=>{
 const r=repetitionCodeExperiment([O,O]);
 assert.deepEqual(r.logicalXProbability,q(parse('7/250')));
 assert.deepEqual(r.inputSquaredFidelity.real,['1','0']);assert.deepEqual(r.entanglementFidelity.real,['243/250','0']);
 assert.equal(r.branches.filter(b=>b.conditionalLogicalDensity===null).length,3);
});
for(const[a,b]of [['4/5','3/5'],['12/13','5/13'],['24/25','7/25'],['1','0']])test(`explicit Bell dilution: ${a},${b}`,()=>{
 const r=bellPairDilution(parse(a),parse(b));
 const wireZero={real:['0','0'],imaginary:['0','0']},wireOne={real:['1','0'],imaginary:['0','0']};
 assert.deepEqual(r.measurementCompleteness,[[wireOne,wireZero],[wireZero,wireOne]]);
 for(const branch of r.branches){assert.deepEqual(branch.probability,['1/2','0']);assert.deepEqual(branch.correctedDensity,r.targetDensity);}
 assert.deepEqual(r.branches[0].corrected,r.branches[1].corrected);
});
test('malformed spectra, errors, probabilities and dilution amplitudes fail',()=>{
 for(const x of [[parse('1/2')],[parse('-1'),parse('2')],new Array(2)])assert.throws(()=>pureStateMajorization(x,[parse('1')]),RangeError);
 for(const errors of [[],['I'],['toString'],new Array(2),Array(17).fill('III')])assert.throws(()=>repetitionKnillLaflamme(errors),RangeError);
 assert.throws(()=>repetitionCodeExperiment([O,Z],'IIII'),RangeError);
 assert.throws(()=>repetitionCodeExperiment([O,Z],'III',parse('3/2')),RangeError);
 assert.throws(()=>bellPairDilution(parse('1/2'),parse('1/2')),RangeError);
});
for(const id of ['quantum-error-correction-syndrome-knill-laflamme','pure-state-entanglement-majorization-resources'])test(`bilingual resource chapter: ${id}`,async()=>{
 const r=await getLearningContent(id);
 for(const locale of ['ko','en']as const){assert.equal(r.outline[locale].length,10);assert(!r[locale].includes('katex-error'));assert(r[locale].includes('<table'));}
 assert(!/[가-힣]/.test(readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`,'utf8')));
});
