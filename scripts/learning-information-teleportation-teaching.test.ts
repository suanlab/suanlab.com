import test from 'node:test';
import assert from 'node:assert/strict';
import {teleportationExamples as examples,readTeleportationDraft,teleportationStateDraft} from '../src/lib/learning-information-teleportation-input';
import {teleportationSteps as steps,teleportationExperiments as guides,teleportationQuestions as questions} from '../src/data/information-teleportation-teaching';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {build} from 'esbuild';
const result=(id:string)=>{const p=examples.find(p=>p.id===id)!;const r=calculate({version:1,requestId:1,input:p.input});assert(r.ok&&r.kind==='teleportation');return r.result;};
for(const p of examples)test(`teleportation teaching ${p.id} independently validates`,()=>{const response=calculate({version:1,requestId:1,input:p.input});assert(response.ok&&response.kind==='teleportation');assert.equal(validate({version:1,requestId:1,input:p.input,response}).phase,'validated');});
test('all branches recover the exact input density with probability one quarter',()=>{
 for(const p of examples){const r=result(p.id).teleportation;assert.equal(r.prefixes.length,5);for(const b of r.branches){assert.deepEqual(b.probability,['1/4','0']);assert.deepEqual(b.correctedDensity,r.inputDensity);}assert.deepEqual(r.branches.map(b=>b.correctionGates.map(g=>g.gate)),[[],['X'],['Z'],['X','Z']]);}
});
test('worked complex amplitudes preserve the branch signs and one-half input scale',()=>{
 const branches=result('complex-input').teleportation.branches;
 const z=(real:string,imaginary='0')=>({real:[real,'0'],imaginary:[imaginary,'0']});
 assert.deepEqual(branches.map(b=>b.rawBob),[[z('3/10'),z('0','2/5')],[z('0','2/5'),z('3/10')],[z('3/10'),z('0','-2/5')],[z('0','-2/5'),z('3/10')]]);
 for(const b of branches)assert.deepEqual(b.correctedBob,[z('3/10'),z('0','2/5')]);
});
test('unconditional Bob states remain maximally mixed for every input and message',()=>{
 const z=(real:string)=>({real:[real,'0'],imaginary:['0','0']});const mixed=[[z('1/2'),z('0')],[z('0'),z('1/2')]];
 for(const p of examples){const r=result(p.id);assert.deepEqual(r.teleportation.bobBeforeMessage,mixed);assert.deepEqual(r.denseCoding.bobBeforeTransmission,mixed);}
});
test('four messages decode exactly and the reversed11 order changes only global sign',()=>{
 for(const[id,index]of [['message-00',0],['message-01',1],['message-10',2],['complex-input',3]]as const){const d=result(id).denseCoding;assert.equal(d.decodedIndex,index);assert.deepEqual(d.decodedProbabilities.map(q=>q[0]),[0,1,2,3].map(i=>i===index?'1':'0'));}
 const a=result('complex-input').denseCoding,b=result('reverse-paulis').denseCoding;assert.equal(a.decodedState[3].real[0],'1');assert.equal(b.decodedState[3].real[0],'-1');assert.deepEqual(a.decodedDensity,b.decodedDensity);assert.deepEqual(a.decodedProbabilities,b.decodedProbabilities);
});
test('raw scale, global phase and irrational norm do not corrupt normalized recovery',()=>{
 const a=result('complex-input').teleportation,b=result('raw-scale').teleportation,c=result('global-phase').teleportation;assert.deepEqual(a.inputDensity,b.inputDensity);assert.deepEqual(a.inputDensity,c.inputDensity);assert.deepEqual(b.inputNormSquared,['25','0']);assert.deepEqual(result('irrational-input').teleportation.inputNormSquared,['4','2']);assert.notDeepEqual(a.input,b.input);
});
test('resource accounting distinguishes classical from quantum transmission',()=>{
 const r=result('complex-input');assert.deepEqual(r.teleportation.resources,{sharedEbits:1,classicalBitsSent:2,quantumInputCopies:1});assert.deepEqual(r.denseCoding.resources,{sharedEbits:1,qubitsSent:1,classicalMessageBits:2});
});
test('drafts preserve all exact coefficients and reject malformed or zero input',()=>{
 for(const p of examples){const x=p.input;assert.deepEqual(readTeleportationDraft(teleportationStateDraft(x.input),String(x.a),String(x.b),x.order),x);}
 const x=teleportationStateDraft(examples[0].input.input);for(const [a,b,order]of [['2','0','x-then-z'],['0','01','x-then-z'],['0','1','other']])assert.equal(readTeleportationDraft(x,a,b,order),null);
 assert.equal(readTeleportationDraft(['0.1',...x.slice(1)],'0','0','x-then-z'),null);assert.equal(readTeleportationDraft(x.slice(1),'0','0','x-then-z'),null);const zero=readTeleportationDraft(Array(8).fill('0'),'0','0','x-then-z');assert(zero);assert.equal(calculate({version:1,requestId:1,input:zero}).ok,false);
});
test('bilingual teaching targets valid examples and explained quiz choices',()=>{assert.equal(examples.length,12);assert.equal(steps.length,7);for(const g of guides){for(const id of g.examples)assert(examples.some(p=>p.id===id));assert(g.predict.ko&&g.predict.en&&g.explanation.ko&&g.explanation.en);}for(const q of questions){assert(q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}});
test('reader imports neither exact circuit execution nor independent validators',async()=>{const r=await build({entryPoints:['src/components/learning/information-teleportation-lab.tsx'],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});assert(!Object.keys(r.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-qubit-circuit-exact|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));});
