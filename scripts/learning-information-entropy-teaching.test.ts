import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {informationEntropyExamples as examples,readInformationEntropyDraft as draft} from '../src/lib/learning-information-entropy-input';
import {informationEntropyExperiments as guides,informationEntropyQuestions as questions} from '../src/data/information-entropy-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='entropy');return r.result;};
for(const p of examples)test(`entropy teaching example independently checked: ${p.id}`,()=>{
 const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok);assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');
});
test('marginal, conditional and entanglement explanations agree with mathematics',()=>{
 for(const[id,joint,conditional,mutual]of [['bell',0,-1,2],['classical-correlation',1,0,1],['independent',2,1,0]]as const){const r=result(id).joint;assert.equal(r.localEntropyA.value,1);assert.equal(r.localEntropyB.value,1);assert.equal(r.jointEntropy.value,joint);assert.equal(r.conditionalEntropy.value,conditional);assert.equal(r.mutualInformation.value,mutual);}
 const positive=result('positive-conditional-entangled').joint;assert(positive.entangled&&positive.conditionalEntropy.value!>0);assert.equal(positive.negativity.value,.125);assert(Math.abs(positive.conditionalEntropy.value!-.5487949406953985)<1e-14);
 const coherent=result('separable-coherence').joint;assert(!coherent.entangled);assert(coherent.dephasingEntropyGain.value!>0);
});
test('rank, retained mass and qubit cost are distinct, including empty space',()=>{
 const three=result('bell').block,two=result('rank-two').block,empty=result('empty-memory').block,pure=result('rank-one-pure').block;
 assert.equal(three.retainedWeight.value,.96);assert.equal(three.requiredQubits,2);assert.equal(two.retainedWeight.value,.8);assert.equal(two.requiredQubits,1);assert.equal(empty.retainedWeight.value,0);assert.equal(empty.requiredQubits,null);assert.equal(pure.retainedWeight.value,1);assert.equal(pure.requiredQubits,0);
});
test('integer precision survives the two memory boundaries',()=>{
 const a=result('rank-before-rounding').block,b=result('rank-after-rounding').block,full=result('full-block-64').block;
 assert.equal(a.retainedRank,'9007199254740992');assert.equal(b.retainedRank,'9007199254740993');assert.equal(a.requiredQubits,53);assert.equal(b.requiredQubits,54);assert.notDeepEqual(a.retainedWeight.bounds,b.retainedWeight.bounds);
 assert.equal(full.dimension,'18446744073709551616');assert.equal(full.retainedRank,full.dimension);assert.equal(full.requiredQubits,64);assert.equal(full.retainedWeight.value,1);
});
test('drafts reject decimal ranks and preserve exact maximum rank',()=>{
 const x=draft('1','1','1/2','64','18446744073709551616');assert(x);assert.equal(x.retainedRank,'18446744073709551616');
 for(const[n,k]of [['64','18446744073709551617'],['2','5'],['0','1'],['65','1'],['2.0','1'],['4','1e1'],['4','01']])assert.equal(draft('1','1','1/2',n,k),null);
 assert.equal(draft('3/2','1','1/2','2','3'),null);assert.equal(draft('1','1','0.5','2','3'),null);
});
test('guides reference real examples and quizzes have explained choices',()=>{
 for(const g of guides){assert(g.examples.length>=2);for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.explanation.ko&&g.explanation.en);}
 for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}
});
test('reader bundle contains no exact mathematical or validation engine',async()=>{
 const r=await build({entryPoints:['src/components/learning/information-entropy-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',metafile:true});
 assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));
});
