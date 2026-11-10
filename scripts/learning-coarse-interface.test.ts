import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {coarseKinds as kinds,coarsePresets as presets,coarsePresetInput as preset,editCoarseField as edit,coarseFields} from '../src/lib/learning-coarse';
import {coarseInputKey as key} from '../src/lib/learning-coarse-protocol';
import {coarseSteps as steps} from '../src/lib/learning-coarse-concepts';
import {coarsePlotPath as path} from '../src/components/learning/coarse-view';
import {evaluateCoarseRequest as evaluate} from '../src/lib/learning-coarse-evaluator';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {CoarseResultView} from '../src/components/learning/coarse-view';
import CoarseConceptPanel from '../src/components/learning/coarse-concept';
import CoarseLab from '../src/components/learning/coarse-lab';
import {LanguageProvider} from '../src/components/language-provider';
for(const [i,kind] of kinds.entries())test(`reader preparations remain independent reference inputs: ${kind}`,()=>{
 const references=JSON.parse(readFileSync(`scripts/fixtures/${['markov-coarse','linear-memory'][i]}-reference.json`,'utf8')).cases;
 assert.equal(steps[kind].length,3);for(const [j,p] of presets[kind].entries()){assert.deepEqual(p.input.parameters,references.find((r:{label:string})=>r.label===p.id).input);assert.equal(key(preset(kind,j)),key(p.input));assert(Object.isFrozen(p.input.parameters));}
 for(const s of steps[kind]){assert(s.presets.length>0);assert(s.presets.every(i=>presets[kind][i]));assert(s.options[s.answer]);assert(s.explanation.ko.length>40&&s.explanation.en.length>60);assert(!/[가-힣]/.test(JSON.stringify(s,(k,v)=>k==='ko'?undefined:v)));}
 const input=preset(kind,0),integerField=i===0?'steps':'timeNumerator',denominatorField=i===0?'transitionDenominator':'timeDenominator';assert(coarseFields(input).every(f=>f.label.ko&&f.label.en));assert.equal(edit(input,integerField,'1e-9999'),null);assert.equal(edit(input,integerField,'1.00000000000000001'),null);assert.equal(edit(input,denominatorField,'0'),null);assert.equal(edit(input,'unknown','1'),null);if(i===0){assert.equal(edit(input,'steps','33'),null);assert.equal(edit(input,'transitionDenominator','0'),null);assert.equal(edit(input,'initialWeights.0','-1'),null);assert(edit(input,'initialWeights.1','2'));assert.equal(edit(input,'initialWeights.3','1'),null);assert(edit(input,'steps','2'));}else{assert.equal(edit(input,'timeNumerator','5'),null);assert(edit(input,'timeNumerator','2'));assert.equal(edit(input,'aNumerator','0'),null);}
});
test('plots retain gaps and use the supplied discrete or physical axis',()=>{
 assert.equal(path([0,1,3,4],[1,null,2,3],x=>x*10,y=>y),'M0 1  M30 2 L40 3');
 assert.equal(path([0,1],[null,null],x=>x,y=>y),' ');
});
test('lab and concept graph does not pull numerical engines onto the main thread',async()=>{
 const b=await build({entryPoints:['src/components/learning/coarse-lab.tsx','src/components/learning/coarse-concept.tsx'],outdir:'unused',format:'esm',bundle:true,write:false,metafile:true,platform:'browser',jsx:'automatic'}),files=Object.keys(b.metafile!.inputs);
 assert(!files.some(p=>/learning-(markov-coarse|linear-memory|linear-memory-math)\.ts|entropy-production-math|interval-transcendentals/.test(p)),files.join('\n'));
});

const result=(family:number,index:number)=>{const response=evaluate({version:1,requestId:1,input:preset(kinds[family],index)});assert(response.ok);return response;};
test('discrete worked predictions agree with the displayed preparations, including signed truncation',()=>{
 const a=result(0,0),b=result(0,1),lift=result(0,3),negative=result(0,7);
 assert(a.kind===kinds[0]&&b.kind===kinds[0]&&lift.kind===kinds[0]&&negative.kind===kinds[0]);
 assert.equal(a.result.history[1].macro[1].value,1/4);assert.equal(b.result.history[1].macro[1].value,3/4);
 assert.equal(lift.result.history[2].macro[1].value,3/16);assert.equal(lift.result.history[2].naiveErrorTV.value,1/16);
 assert.equal(negative.result.history[2].initialAwareTruncated[0].value,5/4);assert.equal(negative.result.history[2].initialAwareTruncated[1].value,-1/4);
 assert.equal(negative.result.history[2].truncatedIsProbability,false);assert.equal(negative.result.history[2].truncatedErrorHalfL1.value,1/4);
});
test('continuous comparisons distinguish initial mismatch, timescales, memory and autonomy',()=>{
 for(const index of [0,3,4]){const r=result(1,index);assert(r.kind===kinds[1]);assert.equal(r.result.markovDriftRate.value,-4/3);assert.equal(r.result.hiddenRelaxationTime.value,[1/3,1/6,1/48][[0,3,4].indexOf(index)]);}
 const quasi=result(1,2),b0=result(1,5),c0=result(1,6),osc=result(1,7);
 assert(quasi.kind===kinds[1]&&b0.kind===kinds[1]&&c0.kind===kinds[1]&&osc.kind===kinds[1]);
 assert(quasi.result.classification.initialQuasiSteadyExactly);assert(quasi.result.history[1].quasiSteadyMismatch.value!==0);
 assert(b0.result.classification.observedStateAutonomousForAllInitialConditions);assert(c0.result.classification.memoryKernelExactlyZero);assert(!c0.result.classification.observedStateAutonomousForAllInitialConditions);assert.notEqual(c0.result.history[0].initialHiddenAnalyticSign,0);assert(osc.result.classification.oscillatoryModes);
});
test('server rendering starts no Worker and preserves bilingual mathematical distinctions',()=>{
 const descriptor=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;
 Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker access');}});
 try{for(const [family,kind]of kinds.entries())for(const language of ['ko','en']as const){
  const response=result(family,family===0?7:8);
  const markup=renderToStaticMarkup(React.createElement(CoarseResultView,{response,language,contextLabel:'SSR',initialRecord:response.result.history.length-1}));
  assert(markup.includes('data-coarse-result'));if(language==='en')assert(!/[가-힣]/.test(markup));
  if(family===0){assert(markup.includes('-0.25'));assert(markup.includes(language==='en'?'not a probability law':'확률 법칙이 아닙니다'));}
  else assert(markup.includes(language==='en'?'Analytic sign':'해석적 부호'));
  assert(renderToStaticMarkup(React.createElement(CoarseConceptPanel,{kind,language})).includes('data-coarse-concept'));
  assert(renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(CoarseLab,{kind}))).includes('data-coarse-lab'));
 }assert.equal(reads,0);}finally{if(descriptor)Object.defineProperty(globalThis,'Worker',descriptor);else Reflect.deleteProperty(globalThis,'Worker');}
});
