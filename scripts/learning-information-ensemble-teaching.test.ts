import test from 'node:test';
import assert from 'node:assert/strict';
import {ensembleExamples as examples,readEnsembleDraft} from '../src/lib/learning-information-ensemble-input';
import {ensembleExperiments as guides,ensembleQuestions as questions} from '../src/data/information-ensemble-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='ensemble');return r.result;};
for(const p of examples)test(`ensemble teaching ${p.id} passes independent validation`,()=>{
 const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='ensemble');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');
});
test('worked comparison keeps preparation fixed while changing measurement',()=>{
 const h=result('helstrom-table'),z=result('z-table');assert.equal(h.holevoInformation.value,z.holevoInformation.value);assert(Math.abs(h.holevoInformation.value!-.7219280948873623)<1e-14);assert(Math.abs(h.selectedMeasurementInformation.value!-.5310044064107188)<1e-14);assert(Math.abs(z.selectedMeasurementInformation.value!-.4330398630967478)<1e-14);
});
test('same average can hide completely different label information',()=>{
 const o=result('orthogonal-z'),s=result('same-mixed');assert.deepEqual(o.averageBloch,s.averageBloch);assert.equal(o.averageStateEntropy.value,1);assert.equal(s.averageStateEntropy.value,1);assert.equal(o.holevoInformation.value,1);assert.equal(s.holevoInformation.value,0);
 for(const id of ['same-pure','certain-label'])assert.equal(result(id).selectedMeasurementInformation.value,0);
});
test('random outcomes can be independent of a perfectly encoded label',()=>{
 const z=result('y-z'),h=result('y-helstrom');assert.equal(z.outputEntropy.value,1);assert.equal(z.conditionalOutputEntropy.value,1);assert.equal(z.selectedMeasurementInformation.value,0);assert.equal(z.holevoInformation.value,1);assert.equal(h.selectedMeasurementInformation.value,1);
});
test('canonical constant guess discards an informative Z record',()=>{
 const c=result('constant-guess'),z=result('biased-z');assert.equal(c.selectedMeasurementInformation.value,0);assert(z.selectedMeasurementInformation.sign===1);assert(Math.abs(z.selectedMeasurementInformation.value!-.010503300578984905)<1e-15);assert.equal(c.holevoInformation.value,z.holevoInformation.value);assert.equal(z.holevoGap.sign,null);assert.equal(z.holevoGap.bounds?.lower.numerator.coefficient,'0');
});
test('tiny information survives rounded equal entropies',()=>{
 const r=result('tiny-z');assert.equal(r.outputEntropy.value,r.conditionalOutputEntropy.value);assert.equal(r.selectedMeasurementInformation.sign,1);assert(Math.abs(r.selectedMeasurementInformation.value!/1.8033688011112042e-61-1)<1e-14);
});
test('guides and bilingual quizzes have valid comparison targets',()=>{
 assert.equal(examples.length,12);assert.equal(new Set(examples.map(p=>p.id)).size,12);
 for(const g of guides){assert(g.examples.length>=2);for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}
 for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}
});
test('drafts retain exact fractions, reject malformed input and defer physicality',()=>{
 const r=['0','0','1'];assert.equal(readEnsembleDraft(r,r,'2/4','z')?.prior,'2/4');assert.equal(readEnsembleDraft(r,r,'1/2','other'),null);assert.equal(readEnsembleDraft(['0.1','0','1'],r,'1/2','z'),null);assert.equal(readEnsembleDraft(r,r,'3/2','z'),null);assert.equal(readEnsembleDraft(['0','0'],r,'1/2','z'),null);
 const physical=readEnsembleDraft(['1','1','1'],r,'1/2','z');assert(physical);assert.equal(calculate({version:1,requestId:1,input:physical}).ok,false);
});
test('ensemble reader excludes calculation and validation engines',async()=>{
 const r=await build({entryPoints:['src/components/learning/information-ensemble-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));
});
