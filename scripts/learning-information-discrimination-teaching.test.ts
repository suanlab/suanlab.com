import test from 'node:test';
import assert from 'node:assert/strict';
import {discriminationExamples as examples,readDiscriminationDraft} from '../src/lib/learning-information-discrimination-input';
import {discriminationExperiments as guides,discriminationQuestions as questions} from '../src/data/information-discrimination-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
for(const p of examples)test(`teaching example ${p.id} uses independently checked mathematics`,()=>{
 const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='discrimination');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');
});
test('worked geometry and biased-prior explanations agree with calculation',()=>{
 const expected:Record<string,[number,number,boolean]>={'rational-pair':[.9,.4,true],orthogonal:[1,.5,true],'complex-y':[1,.5,true],'mixed-pair':[.6,.1,true],'biased-decision':[.9,0,true],'nonunique-threshold':[.6,0,false],identical:[.5,0,false]};
 for(const[id,[success,gain,unique]]of Object.entries(expected)){
  const p=examples.find(p=>p.id===id)!,response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='discrimination');
  assert(Math.abs(response.result.optimalSuccess.value!-success)<1e-14);assert(Math.abs(response.result.gainAbovePriorOnly.value!-gain)<1e-14);assert.equal(response.result.uniqueOptimalEffect,unique);
 }
 const p=examples.find(p=>p.id==='tiny-gain')!,response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='discrimination');assert.equal(response.result.optimalSuccess.value,response.result.priorOnlySuccess.value);assert.equal(response.result.gainAbovePriorOnly.sign,1);assert(Math.abs(response.result.gainAbovePriorOnly.value!/2.5e-31-1)<1e-14);
});
test('every guide comparison and explained quiz has an actionable target',()=>{
 for(const g of guides){assert(g.examples.length>=2);for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}
 for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}
});
test('drafts preserve exact raw fractions and reject malformed coordinates',()=>{
 const a=readDiscriminationDraft(['0','0','1'],['24/25','0','-7/25'],'2/4');assert(a);assert.equal(a.prior,'2/4');assert(Object.isFrozen(a.bloch0));
 for(const [r,p]of [[['0','0','0.1'],'1/2'],[['0','1'],'1/2'],[['0','0','1'],'3/2']]as const)assert.equal(readDiscriminationDraft(r,r,p),null);
});
test('teaching reader has no mathematical computation or validation engine',async()=>{
 const r=await build({entryPoints:['src/components/learning/information-discrimination-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 const inputs=Object.keys(r.metafile!.inputs);assert(!inputs.some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));
});
