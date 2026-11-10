import test from 'node:test';
import assert from 'node:assert/strict';
import {relativeExamples as examples,readRelativeDraft} from '../src/lib/learning-information-relative-input';
import {relativeSteps as steps,relativeExperiments as guides,relativeQuestions as questions} from '../src/data/information-relative-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='relative-entropy');return r.result;};
for(const p of examples)test(`relative teaching ${p.id} passes independent validation`,()=>{
 const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='relative-entropy');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');
});
test('order changes diagonal KL and diagonal processing preserves it',()=>{
 const a=result('diagonal-equality'),b=result('reverse-diagonal');assert.equal(a.before.value?.value,.27807190511263763);assert.equal(b.before.value?.value,.32192809488736235);assert.deepEqual(a.before,a.after);assert.deepEqual(b.before,b.after);
});
test('exact support failure differs from overlap and a nearly pure reference',()=>{
 const a=result('support-failure'),b=result('near-pure'),c=result('equal-pure');assert.equal(a.before.status,'infinite');assert.equal(a.after.value?.value,1);assert.equal(b.before.status,'finite');assert(Math.abs((b.before.value?.value??NaN)-50.32892142331043)<1e-12);assert.equal(b.after.value?.value,1);assert.equal(c.before.reason,'equal-states');assert.equal(c.before.value?.value,0);
});
test('infinity may remain or vanish and is not a floating-point overflow',()=>{
 const o=result('opposite-pure'),m=result('mixed-pure');assert.equal(o.before.status,'infinite');assert.equal(o.after.status,'infinite');assert.equal(o.before.value,null);assert.equal(m.before.status,'infinite');assert.equal(m.after.status,'finite');assert.equal(m.after.value?.value,0);
});
test('the same dephasing erases X and Y differences and reduces the worked example',()=>{
 for(const id of ['erased-pair','complex-y']){const r=result(id);assert.equal(r.before.value?.value,1.2);assert.equal(r.after.value?.value,0);}
 const r=result('noncommuting');assert.equal(r.before.value?.value,1.0150374992788438);assert.equal(r.after.value?.value,.27807190511263763);assert.equal(result('pure-mixed').before.value?.value,1);
});
test('reference-basis weights are not actual-state eigenvalues',()=>{
 const r=result('noncommuting');assert('spectralWeights'in r.before);assert.deepEqual(r.before.spectralWeights?.map(q=>q.value),[.5,.5]);assert.deepEqual(r.before.sigmaSpectrum?.map(q=>q.value),[.9,.1]);
});
test('tiny relative entropy remains positive when reference eigenvalues round equal',()=>{
 const r=result('tiny-finite');assert.equal(r.before.value?.sign,1);assert(Math.abs((r.before.value?.value??NaN)/7.213475204444817e-61-1)<1e-14);assert.equal(r.before.sigmaSpectrum?.[0].value,r.before.sigmaSpectrum?.[1].value);assert.deepEqual(r.before,r.after);
});
test('all bilingual guides target valid examples and quizzes explain their answers',()=>{
 assert.equal(examples.length,13);assert.equal(new Set(examples.map(p=>p.id)).size,13);assert.equal(steps.length,7);
 for(const g of guides){assert(g.examples.length>=2);for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}
 for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}
});
test('drafts reject malformed fractions and defer physicality to the Worker',()=>{
 const r=readRelativeDraft(['0','0','2/4'],['0','0','0']);assert(r);assert.equal(r.rho[2],'2/4');assert(Object.isFrozen(r.rho));assert.equal(readRelativeDraft(['0.1','0','0'],['0','0','0']),null);assert.equal(readRelativeDraft(['0','0'],['0','0','0']),null);
 const p=readRelativeDraft(['1','1','0'],['0','0','0']);assert(p);assert.equal(calculate({version:1,requestId:1,input:p}).ok,false);
});
test('reader excludes numerical and validation engines',async()=>{
 const r=await build({entryPoints:['src/components/learning/information-relative-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));
});
