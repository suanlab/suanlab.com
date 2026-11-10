import test from 'node:test';
import assert from 'node:assert/strict';
import {bb84Examples as examples,readBb84Draft,informationStateDraft} from '../src/lib/learning-information-bb84-input';
import {bb84Steps as steps,bb84Experiments as guides,bb84Questions as questions} from '../src/data/information-bb84-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='bb84');return r.result;};
for(const p of examples)test(`BB84 teaching ${p.id} passes independent validation`,()=>{const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='bb84');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');});
test('CNOT copies basis states but fails on the superposition and complex phase',()=>{
 for(const id of ['basis-zero','basis-one']){const r=result(id);assert(r.cloning.isExactProductClone);assert.equal(r.cloning.singleCopySquaredFidelity.real[0],'1');}
 const plus=result('full-intercept'),complex=result('complex-input');assert.equal(plus.cloning.isExactProductClone,false);assert.equal(plus.cloning.singleCopySquaredFidelity.real[0],'1/2');assert.equal(complex.cloning.singleCopySquaredFidelity.real[0],'337/625');assert.deepEqual(complex.cloning,result('raw-scale').cloning);
});
test('overlap necessary condition is distinct from candidate success',()=>{
 const r=result('identical-rays');assert(r.overlap.compatibleWithClosedUnitaryCloning);assert.equal(r.overlap.squaredOverlap.real[0],'1');assert.equal(r.cloning.isExactProductClone,false);assert.equal(result('full-intercept').overlap.compatibleWithClosedUnitaryCloning,false);assert.equal(result('basis-zero').overlap.squaredOverlap.real[0],'0');
});
test('aggregate QBER hides opposite basis biases',()=>{
 const z=result('biased-eve').attack,x=result('eve-x').attack;assert.equal(z.siftedQber.value,.25);assert.equal(x.siftedQber.value,.25);assert.equal(z.basisQber.z.value,0);assert.equal(z.basisQber.x.value,.5);assert.equal(x.basisQber.z.value,.5);assert.equal(x.basisQber.x.value,0);
});
test('sifting, Eve information, guessing and sample errors use the stated denominators',()=>{
 const r=result('partial-intercept').attack;assert.equal(r.totalProbability.value,1);assert.equal(r.siftProbability.value,.5);assert.equal(r.siftedQber.value,.1);assert.equal(r.eveIdealInformationBits.value,.2);assert.equal(r.eveGuessSuccess.value,.6);assert(Math.abs(r.probabilityAtLeastOneSampleError.value!-(1-.9**20))<1e-14);assert.equal(r.rows.length,80);assert.equal(r.rows.filter(x=>x.aliceBasis===x.bobBasis).length,40);
});
test('zero attack and zero sample are different and tiny detection stays positive',()=>{
 assert.equal(result('no-attack').attack.siftedQber.value,0);const n=result('no-sample').attack;assert.equal(n.siftedQber.value,.25);assert.equal(n.probabilityAtLeastOneSampleError.value,0);const t=result('tiny-attack').attack;assert.equal(t.probabilityAtLeastOneSampleError.sign,1);assert(Math.abs(t.probabilityAtLeastOneSampleError.value!/6.4e-29-1)<1e-14);assert.equal(t.eveGuessSuccess.value,.5);assert.equal(t.eveIdealInformationBits.sign,1);
});
test('state drafts preserve rational/root/complex coefficients independently from BB84 controls',()=>{
 for(const p of examples){const x=p.input;assert.deepEqual(readBb84Draft(informationStateDraft(x.input),informationStateDraft(x.comparison),x.attackFraction,x.eveZProbability,String(x.sampleSize)),x);}
 const p=examples[0].input,a=informationStateDraft(p.input),b=informationStateDraft(p.comparison);for(const sample of ['-1','257','2.0','01'])assert.equal(readBb84Draft(a,b,'1','1/2',sample),null);assert.equal(readBb84Draft(a,b,'1.0','1/2','20'),null);assert.equal(readBb84Draft(a.slice(1),b,'1','1/2','20'),null);
 const zero=readBb84Draft(Array(8).fill('0'),b,'1','1/2','20');assert(zero);assert.equal(calculate({version:1,requestId:1,input:zero}).ok,false);
});
test('every bilingual guide and quiz has an actionable comparison',()=>{assert.equal(examples.length,13);assert.equal(steps.length,7);for(const g of guides){for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}});
test('BB84 reader excludes exact engines and independent validators',async()=>{const r=await build({entryPoints:['src/components/learning/information-bb84-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));});
