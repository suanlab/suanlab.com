import test from 'node:test';
import assert from 'node:assert/strict';
import {majorizationExamples as examples,majorizationDraft,readMajorizationDraft} from '../src/lib/learning-information-majorization-input';
import {majorizationSteps as steps,majorizationExperiments as guides,majorizationQuestions as questions} from '../src/data/information-majorization-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='majorization');return r.result;};
const compare=(id:string)=>{const r=result(id);assert(r.mode==='compare');return r.comparison;};
for(const p of examples)test(`majorization teaching ${p.id} independently validates`,()=>{const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='majorization');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');});
test('crossing prefixes forbid both directions despite one entropy decrease',()=>{
 const r=compare('incomparable');assert.equal(r.relation,'incomparable');assert.equal(r.forward.firstViolation,1);assert.equal(r.reverse.firstViolation,2);assert.equal(r.forward.optimalStochasticSuccess.value,5/6);assert.equal(r.reverse.optimalStochasticSuccess.value,4/5);assert(r.targetEntropy.value!>r.sourceEntropy.value!);assert.equal(r.forward.tails[2].ratio?.value,1.25);
});
test('Bell/product direction and zero-tail constraints match the resource obstruction',()=>{
 const b=compare('bell-product'),p=compare('rank-obstruction');assert.equal(b.forward.deterministic,true);assert.equal(b.reverse.optimalStochasticSuccess.value,0);assert.equal(p.forward.deterministic,false);assert.equal(p.forward.optimalStochasticSuccess.value,0);assert.equal(b.forward.tails[1].applicable,false);assert.equal(b.forward.tails[1].ratio,null);
});
test('sorting is stable and explicit zeros differ from padding',()=>{
 const r=compare('permuted-equal');assert.equal(r.relation,'equivalent');assert.deepEqual(r.sourceSortingIndices,[1,0,2]);assert.deepEqual(r.targetSortingIndices,[0,1,2]);const z=compare('zero-padded');assert.equal(z.dimension,3);assert.equal(z.sourceSortingIndices.length,3);assert.equal(z.targetSortingIndices.length,1);assert.equal(z.sourceSchmidtRank,2);assert.equal(z.targetSchmidtRank,1);
 const u=compare('uniform-sixteen');assert.equal(u.dimension,16);assert.equal(u.sourceEntropy.value,4);assert.equal(u.forward.deterministic,true);assert.equal(u.forward.tails.filter(x=>!x.applicable).length,15);
});
test('a rounded success of one does not hide an exact prefix violation',()=>{
 const r=compare('tiny-violation');assert.equal(r.forward.optimalStochasticSuccess.value,1);assert.equal(r.forward.deterministic,false);assert.equal(r.forward.prefixes[0].targetMinusSource.sign,-1);assert(Math.abs(r.forward.prefixes[0].targetMinusSource.value!/-1e-30-1)<1e-14);assert.equal(r.source[0].value,r.target[0].value);
 const t=compare('tiny-success');assert.equal(t.forward.optimalStochasticSuccess.sign,1);assert.equal(t.forward.optimalStochasticSuccess.value,2e-30);assert.equal(t.sourceEntropy.sign,1);
});
test('explicit dilution branches and endpoints reach their target with probability one half',()=>{
 for(const id of ['actual-dilution','swapped-amplitudes','bell-endpoint-zero','bell-endpoint-one']){const r=result(id);assert(r.mode==='bell-dilution');assert.deepEqual(r.dilution.measurementCompleteness,[[{real:['1','0'],imaginary:['0','0']},{real:['0','0'],imaginary:['0','0']}],[{real:['0','0'],imaginary:['0','0']},{real:['1','0'],imaginary:['0','0']}]]);for(const b of r.dilution.branches){assert.deepEqual(b.probability,['1/2','0']);assert.equal(b.bothPartiesApplyX,b.outcome===1);assert.deepEqual(b.correctedDensity,r.dilution.targetDensity);}}
 const c=compare('bell-target-comparison');assert.equal(c.forward.deterministic,true);assert.equal(c.reverse.optimalStochasticSuccess.value,.72);
});
test('mode drafts preserve probabilities/amplitudes without silently normalizing',()=>{
 for(const p of examples)assert.deepEqual(readMajorizationDraft(majorizationDraft(p.input)),p.input);
 const d={mode:'compare' as const,source:'1/2, 1/4 1/4',target:'2/5 2/5 1/5',a:'4/5',b:'3/5'};assert(readMajorizationDraft(d));assert.equal(readMajorizationDraft({...d,source:'0.5,1/2'}),null);assert.equal(readMajorizationDraft({...d,source:Array(17).fill('0').join(',')}),null);assert.equal(readMajorizationDraft({...d,target:''}),null);
 const wrongSum=readMajorizationDraft({...d,source:'1/2'});assert(wrongSum);assert.equal(calculate({version:1,requestId:1,input:wrongSum}).ok,false);
 const wrongNorm=readMajorizationDraft({...d,mode:'bell-dilution',a:'1/2',b:'1/2'});assert(wrongNorm);assert.equal(calculate({version:1,requestId:1,input:wrongNorm}).ok,false);
});
test('bilingual explanations and quizzes point to valid comparisons',()=>{assert.equal(examples.length,14);assert.equal(steps.length,7);for(const g of guides){for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}});
test('reader excludes mathematical engines and independent validation',async()=>{const r=await build({entryPoints:['src/components/learning/information-majorization-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-qubit-circuit-exact|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));});
