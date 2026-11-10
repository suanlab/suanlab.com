import test from 'node:test';
import assert from 'node:assert/strict';
import {capacityExamples as examples,readCapacityDraft} from '../src/lib/learning-information-capacity-input';
import {capacitySteps as steps,capacityExperiments as guides,capacityQuestions as questions} from '../src/data/information-capacity-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='capacity');return r.result;};
for(const p of examples)test(`capacity teaching ${p.id} independently validates`,()=>{const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='capacity');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');});
test('family capacities stay fixed while selected-input information changes',()=>{
 const mixed=result('phase-noise');for(const id of ['pure-input','pure-z','mixed-z','mixed-y'])assert.deepEqual(result(id).capacitiesByFamilyTheorem,mixed.capacitiesByFamilyTheorem);assert.equal(mixed.capacitiesByFamilyTheorem.quantumQubitsPerUse.value,.5310044064107188);assert.equal(result('mixed-z').selectedInputCoherentInformation.value,.38908532390661177);assert.equal(result('mixed-y').selectedInputCoherentInformation.value,.3403926800658453);
});
test('pure-input zero coherent information does not make identity capacity zero',()=>{
 const r=result('pure-identity');assert.equal(r.selectedInputCoherentInformation.value,0);assert.equal(r.selectedInputReferenceMutualInformation.value,0);assert.equal(r.capacitiesByFamilyTheorem.quantumQubitsPerUse.value,1);for(const id of ['pure-input','pure-z'])assert.equal(result(id).selectedInputCoherentInformation.sign,0);
});
test('environment coherence changes its entropy from classical flag entropy',()=>{
 const z=result('mixed-z'),pure=result('pure-z');assert.equal(z.environmentDensity[0][1].real.value,.18);assert.equal(z.environmentEntropy.value,.3328427709807506);assert.notEqual(z.environmentEntropy.value,z.phaseErrorEntropy.value);assert.equal(pure.environmentDensity[0][1].real.value,.3);assert.equal(pure.environmentEntropy.value,0);assert(pure.phaseErrorEntropy.value!>0);
});
test('known unitary reverses coordinates while complete dephasing erases transverse components',()=>{
 const u=result('rotated-unitary'),d=result('mixed-complete');assert.deepEqual(u.outputBloch.map(q=>q.value),[0,-.6,.2]);assert.equal(u.environmentEntropy.value,0);assert.equal(u.inputEntropy.value,u.selectedInputCoherentInformation.value);assert.deepEqual(d.outputBloch.map(q=>q.value),[0,0,.2]);assert.equal(d.selectedInputCoherentInformation.value,0);
});
test('complementary p and1-p have equal rates and known Z restores endpoint capacity',()=>{
 assert.deepEqual(result('phase-noise').capacitiesByFamilyTheorem,result('complementary-noise').capacitiesByFamilyTheorem);assert.deepEqual(result('identity').capacitiesByFamilyTheorem,result('known-z').capacitiesByFamilyTheorem);const q=result('complete-dephasing').capacitiesByFamilyTheorem;assert.equal(q.classicalBitsPerUse.value,1);assert.equal(q.entanglementAssistedClassicalBitsPerUse.value,1);assert.equal(q.quantumQubitsPerUse.value,0);
});
test('tiny positive Q survives rounded entropy one on both sides of1/2',()=>{
 const a=result('tiny-capacity'),b=result('tiny-below');assert.equal(a.phaseErrorEntropy.value,1);assert.equal(a.capacitiesByFamilyTheorem.quantumQubitsPerUse.sign,1);assert(a.quantumCapacityStrictlyPositive);assert.equal(a.capacitiesByFamilyTheorem.quantumQubitsPerUse.value,b.capacitiesByFamilyTheorem.quantumQubitsPerUse.value);assert(Math.abs(a.capacitiesByFamilyTheorem.quantumQubitsPerUse.value!/7.213475204444817e-61-1)<1e-14);assert.equal(result('complete-dephasing').quantumCapacityStrictlyPositive,false);
});
test('classical information and guessing success have distinct values',()=>{
 const r=result('phase-noise'),d=result('complete-dephasing');assert.equal(r.zBasisMessageInformation.value,1);assert.equal(r.xBasisMessageInformation.value,.5310044064107188);assert.equal(r.xBasisEqualPriorGuessSuccess.value,.9);assert.equal(d.xBasisMessageInformation.value,0);assert.equal(d.xBasisEqualPriorGuessSuccess.value,.5);
});
test('drafts preserve exact raw p and reject formats or nonphysical vectors appropriately',()=>{
 const r=readCapacityDraft(['0','0','2/4'],'2/4');assert(r);assert.equal(r.input[2],'2/4');assert.equal(r.probability,'2/4');assert(Object.isFrozen(r.input));assert.equal(readCapacityDraft(['0','0','0.1'],'1/10'),null);assert.equal(readCapacityDraft(['0','0'],'1/10'),null);assert.equal(readCapacityDraft(['0','0','0'],'3/2'),null);
 const physical=readCapacityDraft(['1','1','0'],'1/10');assert(physical);assert.equal(calculate({version:1,requestId:1,input:physical}).ok,false);
});
test('bilingual guides and explained quizzes refer to real examples',()=>{assert.equal(examples.length,14);assert.equal(steps.length,8);for(const g of guides){for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}});
test('capacity reader excludes numerical models and independent validators',async()=>{const r=await build({entryPoints:['src/components/learning/information-capacity-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-qubit-circuit-exact|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));});
