import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test,{after} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {bosonTeachingPresets as presets,bosonTeachingInput as preset,bosonBasisTrace,editBosonTeachingField as edit,bosonTeachingFields as fields} from '../src/lib/learning-many-body-boson-teaching';
import {bosonFockModel as raw,createBosonFockParameters as create} from '../src/lib/learning-boson-fock';
import {bosonCoherentModel as coherent} from '../src/lib/learning-boson-coherent';
import {manyBodyInputKey,parseManyBodyStoredInput} from '../src/lib/learning-many-body-input';
import {manyBodyBosonUnit as unit,manyBodyBosonGuide as guide} from '../src/data/many-body-boson-teaching';
import {getLearningContent} from '../src/lib/learning-content';
ensureTestArtifactDirectory('.runtime/learning/many-body-boson-teaching', { recursive: true });

const runtime=process.env.BOSON_TEACHING_RUNTIME??'.runtime/learning/many-body-boson-teaching';let traceChecks=0;
test('six raw preparations match independent hand calculations of energy,variance and boundary CCR',()=>{
 const expected=[[0,0,1],[3,0,-3],[1.5,2.25,-1],[0,0,0],[-3,0,-3],[0,1,1]];
 presets.slice(0,6).forEach((p,i)=>{const r=raw(p.input.parameters);assert.deepEqual([r.energy.observation.value,r.variance.observation.value,r.localStates[0].finiteCommutatorExpectation.observation.value],expected[i]);});
 const top=raw(preset(1).parameters);assert.equal(top.localStates[0].quadratures.robertsonBound.observation.value,2.25);assert.equal(top.localStates[0].quadratures.uncertaintySlack.exactSign,0);assert.equal(raw(preset(4).parameters).classification.selectedStateFiniteCutoffGround,true);assert.equal(raw(preset(5).parameters).totalNumber.conserved,true);
});
test('coherent cutoff1 and3 keep distinct infinite,projected and normalized state accounts',()=>{
 for(const[index,mean,top,comm,series]of [[6,.5,.5,0,2],[7,15/16,1/16,.75,8/3]]){const r=coherent(preset(index).parameters);assert.equal(r.infiniteState.meanTotalNumber.value,1);assert.equal(r.normalizedProjection.totalNumber.expectation.observation.value,mean);assert.equal(r.perMode[0].normalizedProjection.topProbability.value,top);assert.equal(r.perMode[0].normalizedProjection.finiteCommutatorExpectation.value,comm);assert(Math.abs(r.rawSeriesNorm.value!-series)<1e-14);assert(Math.abs(r.truncation.retainedProbability.value!-series/Math.E)<1e-14);assert.deepEqual(r.unnormalizedProjection.normSquared,r.normalizedProjection.fidelitySquaredToInfinite);assert.deepEqual(r.unnormalizedProjection.distanceSquaredToInfinite,r.truncation.discardedProbability);assert.equal(r.perMode[0].normalizedProjection.annihilationEigenResidualSquared.observation.value,top);}
 const imaginary=coherent(preset(8).parameters);assert.equal(imaginary.perMode[0].infiniteState.xMean.exactSign,0);assert(Math.abs(imaginary.perMode[0].infiniteState.pMean.observation.value!-Math.SQRT2)<1e-14);
});
test('tiny positive tails and rectangular two-mode cutoffs retain their meaning',()=>{
 const small=coherent(preset(10).parameters);assert.equal(small.truncation.discardedProbability.sign,1);assert(small.truncation.discardedProbability.value!>0);assert(Math.abs(small.truncation.discardedProbability.value!/(1e-48/24)-1)<1e-10);
 const large=coherent(preset(9).parameters);assert(large.truncation.discardedProbability.value!>.99);
 const two=coherent(preset(11).parameters);assert(Math.abs(two.truncation.retainedProbability.value!-5/Math.E**2)<1e-14);assert.equal(two.normalizedProjection.totalNumber.expectation.observation.value,1.3);assert.deepEqual(two.cutoffs,[1,2]);
});
test('every basis and two-action trace across20 cutoff profiles matches an independent occupation oracle',()=>{
 for(const modes of[1,2])for(let k0=0;k0<=3;k0++)for(let k1=0;k1<=(modes===2?3:0);k1++){
  const p=create(modes,k0,k1),r=raw(p);for(let basis=0;basis<r.dimension;basis++)for(let left=0;left<modes;left++)for(let right=0;right<modes;right++)for(const lc of[0,1])for(const rc of[0,1])for(const reverse of[false,true]){
   const parameters={...p,leftMode:left,rightMode:right,leftCreation:lc,rightCreation:rc},trace=bosonBasisTrace(r,parameters,basis,reverse);let index:number|null=basis,factor=1;
   for(const[i,[mode,creation]]of(reverse?[[left,lc],[right,rc]]:[[right,rc],[left,lc]]).entries()){
    if(index!==null){const n:number[]=modes===1?[index]:[Math.floor(index/(k1+1)),index%(k1+1)],cut=mode===0?k0:k1;if((creation&&n[mode]===cut)||(!creation&&n[mode]===0)){index=null;factor=0;}else{factor*=creation?n[mode]+1:n[mode];n[mode]+=creation?1:-1;index=modes===1?n[0]:n[0]*(k1+1)+n[1];}}
    assert.equal(trace[i+1].basis,index);assert.equal(trace[i+1].squaredCoefficient,factor);
   }traceChecks++;
  }
 }assert(traceChecks>3000);
 const r=raw(preset().parameters);assert.equal(bosonBasisTrace(r,{...r.parameters,leftCreation:1,rightCreation:1},0)[2].squaredCoefficient,2);assert.equal(bosonBasisTrace(r,r.parameters,3)[1].basis,null);assert.throws(()=>bosonBasisTrace(r,r.parameters,4));
});
test('editors reject invalid lexical,zero norm and out-of-range coherent inputs without mutation',()=>{
 const p=preset();for(const value of['','-0','1.0','1e0','+1','01','1000001'])assert.equal(edit(p,'onsite0',value),null);assert.equal(edit(p,'stateReal0','0'),null);assert.equal(edit(p,'hDenominator','0'),null);assert.equal(edit(preset(6),'alphaReal0','5'),null);assert.equal(edit(preset(6),'alphaDenominator0','0'),null);assert.equal(edit(p,'madeUp','1'),null);assert.equal(edit(p,'onsite0','-2')?.parameters.onsite0,-2);assert.equal(p.parameters.onsite0,1);assert.throws(()=>preset(12));
});
test('both models round-trip owned storage independently with active mixed-radix labels',()=>{
 presets.forEach((p,i)=>{assert.equal(Object.keys(p.input.parameters).length,i<6?52:26);assert.equal(manyBodyInputKey(parseManyBodyStoredInput(JSON.stringify({version:1,value:p.input}))!),manyBodyInputKey(preset(i)));});assert.equal(parseManyBodyStoredInput(JSON.stringify({version:2,value:preset()})),null);assert(fields(preset(5)).find(v=>v.key==='stateReal2')!.label.en.includes('|1,0〉'));assert(!fields(preset(6)).some(f=>f.key.startsWith('state')));
});
test('three concepts,guides and explained quizzes accompany unchanged bilingual audited chapters',async()=>{
 assert.equal(guide.steps.length,3);assert.equal(guide.experiments.length,3);assert.deepEqual(unit.questions.map(q=>q.answer),[1,0,2]);const content=await getLearningContent(unit.id);for(const lang of['ko','en'] as const){assert.equal(readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`,'utf8'),readFileSync(`docs/online-learning/physics-math-for-ai/${unit.id}-chapter.${lang}.md`,'utf8'));assert.equal(content.outline[lang].length,18);assert.equal((content[lang].match(/<table/g)||[]).length,4);assert(!content[lang].includes('katex-error'));for(const q of unit.questions)assert(q.explanation[lang]);}
});
test('main lesson bundle excludes calculators and mathematical validators',async()=>{const b=await build({entryPoints:['src/components/learning/many-body-boson-lesson.tsx'],jsx:'automatic',bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(b.metafile!.inputs);assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|boson-algebraic|tensor-roots|quantum-hilbert-core|observable-matrix|ritz-interval|interval-transcendentals|many-body-(?:evaluator|validation-evaluator|.*-validation))\.ts$/.test(p)),inputs.join('\n'));writeFileSync(`${runtime}/main-thread-dependency-graph.json`,JSON.stringify({inputs,heavyNumericalEnginesExcluded:true,mathValidatorsExcluded:true},null,2)+'\n');});
after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({traceChecks,fixedPreparations:12,conceptSteps:3,guidedExperiments:3,explainedQuestions:3,bilingualManuscripts:2},null,2)+'\n'));
