import test from 'node:test';
import assert from 'node:assert/strict';
import {errorCorrectionExamples as examples,readErrorCorrectionDraft,errorCorrectionStateDraft} from '../src/lib/learning-information-error-correction-input';
import {errorCorrectionSteps as steps,errorCorrectionExperiments as guides,errorCorrectionQuestions as questions} from '../src/data/information-error-correction-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='error-correction');return r.result;};
for(const p of examples)test(`error-correction teaching ${p.id} independently validates`,()=>{const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='error-correction');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');});
test('single X sites have the expected syndrome and recover complex coherence',()=>{
 for(const[id,s]of [['single-x','10'],['middle-x','11'],['last-x','01']]){const r=result(id).experiment,b=r.branches.find(b=>b.conditionalLogicalDensity!==null)!;assert.equal(b.syndrome.join(''),s);assert.equal(b.conditionalLogicalDensity?.[0][1].imaginary[0],'-12/25');assert.deepEqual(b.rawRecovered,r.input);assert.equal(r.branches.filter(b=>b.conditionalLogicalDensity===null).length,3);for(const b of r.branches)if(b.conditionalLogicalDensity===null){assert.deepEqual(b.probability,['0','0']);assert(b.rawRecovered.every(z=>z.real.every(x=>x==='0')&&z.imaginary.every(x=>x==='0')));}}
});
test('phase and double X errors leave different logical operators under the fixed recovery',()=>{
 const z=result('phase-failure').experiment.branches[0],x=result('double-x').experiment.branches[1];assert.equal(z.logicalOperator[1][1].real[0],'-1');assert.equal(z.conditionalLogicalDensity?.[0][1].imaginary[0],'12/25');assert.equal(x.logicalOperator[0][1].real[0],'1');assert.equal(x.logicalOperator[1][0].real[0],'1');
});
test('KL passes a known or degenerate Z family but not identity plus Z',()=>{
 for(const id of ['known-z','degenerate-family']){const r=result(id);assert(r.condition.conditionHolds);assert.equal(r.experiment.branches[0].logicalOperator[1][1].real[0],'-1');}
 for(const id of ['identity-added','phase-failure']){const r=result(id).condition;assert.equal(r.conditionHolds,false);assert.deepEqual(r.firstWitness?.errors,['III','ZII']);assert(r.pairs.filter(p=>p.a===p.b).every(p=>p.scalarOnCode));}
 assert.equal(result('maximum-family').condition.pairs.length,256);
});
test('IID noise and selected Pauli branches remain independent experiments',()=>{
 const z=result('zero-noise').experiment;assert.equal(z.logicalXProbability.value,0);assert.equal(z.entanglementFidelity.real[0],'1');assert.equal(z.branches[0].logicalOperator[1][1].real[0],'-1');
 const a=result('single-x').experiment,b=result('double-x').experiment;assert.deepEqual(a.logicalOutput,b.logicalOutput);assert.equal(a.logicalXProbability.value,.028);assert.equal(a.entanglementFidelity.real[0],'243/250');
});
test('one-input fidelity can miss reference entanglement damage',()=>{const r=result('misleading-plus').experiment;assert.equal(r.inputSquaredFidelity.real[0],'1');assert.equal(r.entanglementFidelity.real[0],'243/250');assert.equal(r.logicalXProbability.value,.028);});
test('tiny and high-noise probabilities retain exact classifications and scale',()=>{
 const t=result('tiny-noise').experiment;assert.equal(t.logicalXProbability.sign,1);assert(Math.abs(t.logicalXProbability.value!/3e-60-1)<1e-14);assert.equal(result('high-noise').experiment.logicalXProbability.value,27/32);assert.equal(result('complete-noise').experiment.logicalXProbability.value,1);
});
test('exact drafts support duplicates and roots while rejecting malformed families',()=>{
 for(const p of examples){const x=p.input;assert.deepEqual(readErrorCorrectionDraft(errorCorrectionStateDraft(x.input),x.selectedError,x.physicalXProbability,x.errors.join(', ')),x);}
 const input=errorCorrectionStateDraft(examples[0].input.input),root=[...input];root[1]='1/2';const accepted=readErrorCorrectionDraft(root,'XII','1/10','III XII,IXI IIX');assert(accepted);assert.equal(accepted.input[0].real[1],'1/2');const response=calculate({version:1,requestId:1,input:accepted});assert(response.ok);assert.equal(validate({version:1,requestId:1,input:accepted,response}).phase,'validated');
 for(const errors of ['',Array(17).fill('III').join(','),'IIII','xii','III;XII'])assert.equal(readErrorCorrectionDraft(input,'XII','1/10',errors),null);assert.equal(readErrorCorrectionDraft(input,'ABC','1/10','III'),null);assert.equal(readErrorCorrectionDraft(input,'III','2','III'),null);const zero=readErrorCorrectionDraft(Array(8).fill('0'),'III','0','III');assert(zero);assert.equal(calculate({version:1,requestId:1,input:zero}).ok,false);
});
test('all bilingual guides and quizzes have valid explanatory targets',()=>{assert.equal(examples.length,15);assert.equal(steps.length,7);for(const g of guides){for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}});
test('error-correction reader excludes numerical engines and independent validators',async()=>{const r=await build({entryPoints:['src/components/learning/information-error-correction-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-qubit-circuit-exact|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));});
