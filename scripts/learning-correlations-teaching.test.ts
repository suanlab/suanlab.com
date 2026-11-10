import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {readFileSync} from 'node:fs';
import {correlationKinds as kinds,correlationPresets as presets,correlationGuides as guides,correlationPresetInput as preset,parseCorrelationNumber as parse} from '../src/lib/learning-correlations';
import {correlationConceptSteps as steps,correlationConceptInputs as inputs,correlationConceptQuestions as questions} from '../src/lib/learning-correlations-concepts';
import {isCorrelationInput,correlationInputKey as key,type CorrelationInput} from '../src/lib/learning-correlations-protocol';
import {evaluateCorrelationRequest as evaluate} from '../src/lib/learning-correlations-evaluator';
import {CorrelationResultView,correlationQuantity,correlationMagnitudeLog} from '../src/components/learning/correlations-view';
import CorrelationsLab from '../src/components/learning/correlations-lab';
import CorrelationConceptPanel from '../src/components/learning/correlations-concept';
import {useCorrelationSeries} from '../src/components/learning/use-correlations-series';
import {LanguageProvider} from '../src/components/language-provider';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {build} from 'esbuild';
(globalThis as typeof globalThis&{React?:typeof React}).React=React;
const cache=new Map<string,ReturnType<typeof evaluate>>();
function result(input:CorrelationInput){const k=key(input);if(!cache.has(k))cache.set(k,evaluate({version:1,requestId:1,input}));const r=cache.get(k)!;assert.ok(r.ok);return r;}
for(const kind of kinds)test(`${kind}: substantive guides cover every distinct preset and fixed steps remain owned and independent`,()=>{
 assert.equal(guides[kind].length,3);assert.equal(steps[kind].length,6);assert.equal(questions[kind].length,3);
 assert.deepEqual([...new Set(guides[kind].flatMap(g=>g.presetIndices))].sort((a,b)=>a-b),presets[kind].map((_,i)=>i));assert.equal(new Set(presets[kind].map(x=>x.id)).size,presets[kind].length);
 for(const g of guides[kind])for(const lang of ['ko','en'] as const){for(const field of [g.title,g.predict,g.action,g.explain]){assert.ok(field[lang].trim().length>0);if(lang==='en')assert.doesNotMatch(field[lang],/[가-힣]/);}assert.ok(g.explain[lang].length>g.predict[lang].length);}
 for(const [i,s] of steps[kind].entries()){
  assert.equal(s.check.options.length,3);assert.ok(s.check.answer>=0&&s.check.answer<3);assert.ok(s.samples.length>=1);assert.ok(s.explanation.en.length>150);assert.ok(inputs(kind,i).every(isCorrelationInput));
  const a=inputs(kind,i),b=inputs(kind,i);assert.notEqual(a[0],b[0]);assert.ok(Object.isFrozen(a[0].parameters));
 }
});
for(const kind of kinds)for(const [i,p] of presets[kind].entries())test(`${p.id}: actual experiment result renders bilingual scientific status with honest signs and representation`,()=>{
 const r=result(preset(kind,i));for(const language of ['ko','en'] as const){const html=renderToStaticMarkup(React.createElement(CorrelationResultView,{response:r,language}));assert.ok(html.includes(`data-correlation-result="${kind}"`));assert.ok(html.includes('<table'));assert.doesNotMatch(html,/NaN|undefined|katex-error/);if(language==='en')assert.doesNotMatch(html,/[가-힣]/);}
});
for(const kind of kinds)test(`${kind}: every fixed concept preparation computes its declared model and visual result`,()=>{
 for(let step=0;step<6;step++)for(const input of inputs(kind,step)){const r=result(input);const html=renderToStaticMarkup(React.createElement(CorrelationResultView,{response:r,language:'en'}));assert.ok(html.includes(`data-correlation-result="${input.kind}"`));assert.doesNotMatch(html,/[가-힣]|NaN|undefined/);}
});
test('fixed preparations distinguish bias, source scaling, tiny nonzero values, boundaries and endpoint labels',()=>{
 const biased=result(preset(kinds[0],4));assert.ok(biased.kind===kinds[0]);assert.equal(biased.result.covariance[0][1].resolution,'structural-zero');assert.ok(biased.result.pairMean.value!<0);
 const base=result(preset(kinds[0],5)),scaled=result(preset(kinds[0],6));assert.ok(base.kind===kinds[0]&&scaled.kind===kinds[0]);assert.ok(Math.abs(scaled.result.probeVariance.value!-4*base.result.probeVariance.value!)<1e-12);
 const cap=result(preset(kinds[0],9)),tiny=result(preset(kinds[0],10));assert.ok(cap.kind===kinds[0]&&tiny.kind===kinds[0]);assert.equal(cap.result.covariance[0][1].resolution,'unresolved-at-precision-ceiling');assert.equal(tiny.result.probeVariance.value,null);assert.equal(tiny.result.probeVariance.sign,1);
 const odd=result(preset(kinds[1],4)),even=result(preset(kinds[1],5));assert.ok(odd.kind===kinds[1]&&even.kind===kinds[1]);assert.ok(odd.result.commonFieldResponsePerSpin.value!>.007);assert.ok(even.result.commonFieldResponsePerSpin.value!<1e-10);
 const ar=result(preset(kinds[2],7));assert.ok(ar.kind===kinds[2]);assert.equal(ar.result.meanVariance.resolution,'structural-zero');assert.equal(ar.result.varianceEquivalentSampleCount.resolution,'mathematical-infinity');assert.equal(ar.result.asymptoticIntegratedCorrelationTime,null);
});
test('decimal controls preserve zero boundaries and reject nonzero underflow and malformed text',()=>{
 for(const text of ['1e-9999','NaN','Infinity','1,2','','0x10'])assert.equal(parse(text).ok,false,text);
 assert.deepEqual(parse('5e-324'),{ok:true,value:Number.MIN_VALUE});assert.deepEqual(parse('-0'),{ok:true,value:0});assert.deepEqual(parse('0e-9999'),{ok:true,value:0});
 assert.throws(()=>preset(kinds[0],.5),RangeError);assert.throws(()=>inputs(kinds[1],6),RangeError);
});
test('SSR experiments, concept series and panels never access Worker or execute numerical work',()=>{
 const original=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR worker');}});
 function Probe(){const r=useCorrelationSeries([preset(kinds[0],0)]);assert.equal(r.status,'pending');assert.deepEqual(r.responses,[]);return React.createElement('output',null,'pending');}
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(CorrelationsLab,{kind}),React.createElement(CorrelationConceptPanel,{kind,language:'en'})));assert.ok(html.includes('data-correlation-ready="false"'));assert.ok(html.includes('data-correlation-concept-state="pending"'));assert.ok(!html.includes('data-correlation-result='));}renderToStaticMarkup(React.createElement(Probe));assert.equal(reads,0);}finally{if(original)Object.defineProperty(globalThis,'Worker',original);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('registered teaching exposes three correlation readers and labels missing endpoint values without calling them zero',()=>{
 assert.ok(learningUnits.length>=527);assert.equal(learningUnits.slice(0,527).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1581);const ids:string[]=learningUnits.map(u=>u.id);for(const kind of kinds)assert.equal(ids.includes(kind),true);
 assert.equal(correlationQuantity(null,(_ko,en)=>en),'Not provided for this preparation');
 const r=result(preset(kinds[2],7));const html=renderToStaticMarkup(React.createElement(CorrelationResultView,{response:r,language:'en'}));assert.ok(html.includes('Not provided'));assert.ok(html.includes('+∞'));assert.doesNotMatch(html,/Undefined/);
});
test('rounded-log reference inputs are pinned across server and browser runtimes',()=>{
 const a=preset(kinds[0],1),b=preset(kinds[1],0);assert.ok(a.kind===kinds[0]&&b.kind===kinds[1]);assert.equal(a.parameters.couplingK,.5493061443340549);assert.equal(b.parameters.couplingK,a.parameters.couplingK);
 assert.doesNotMatch(readFileSync('src/lib/learning-correlations.ts','utf8'),/Math\.log\(/);
});
test('exact-zero matrix entries have a neutral legend rather than positive shading',()=>{
 const r=result(preset(kinds[0],0)),html=renderToStaticMarkup(React.createElement(CorrelationResultView,{response:r,language:'en'}));
 assert.ok(html.includes('neutral exact zero'));assert.equal((html.match(/fill="hsl\(var\(--muted\)\)"/g)??[]).length,2);
});
test('displayed logs near one avoid cancellation signs while underflow retains its separate magnitude',()=>{
 const r=result(preset(kinds[2],5));assert.ok(r.kind===kinds[2]);assert.equal(correlationMagnitudeLog(r.result.autocovariances[0].covariance),0);
 for(const row of r.result.autocovariances.slice(1))assert.ok(correlationMagnitudeLog(row.covariance)!<=0);
 const tiny=result(preset(kinds[0],10));assert.ok(tiny.kind===kinds[0]);assert.equal(tiny.result.probeVariance.value,null);assert.equal(correlationMagnitudeLog(tiny.result.probeVariance),tiny.result.probeVariance.logMagnitude);
});
test('teaching components do not import numerical evaluators or model arithmetic into the browser graph',async()=>{
 const r=await build({entryPoints:['src/components/learning/correlations-lab.tsx','src/components/learning/correlations-concept.tsx'],bundle:true,write:false,outdir:'unused',format:'esm',platform:'browser',metafile:true,external:['react'],logLevel:'silent'});
 const paths=Object.keys(r.metafile!.inputs);for(const file of ['learning-correlations-evaluator.ts','learning-binary-correlations.ts','learning-periodic-ising-ring.ts','learning-ar1-average.ts','learning-binary-products.ts','learning-interval-transcendentals.ts'])assert.ok(!paths.some(p=>p.endsWith(file)),file);
});


test('count controls reject fractional decimal lexemes before binary64 erases their fractions',()=>{
 for(const text of ['3.00000000000000001','2.99999999999999999','511.99999999999999999','5.1200000000000000001e2','1e-9999'])assert.deepEqual(parse(text,true),{ok:false},text);
 assert.deepEqual(parse('3.00000000000000001'),{ok:true,value:3});
});
test('count controls retain exact integer decimal notation and each model domain',()=>{
 for(const [text,value]of [['3.000',3],['30e-1',3],['.03e2',3],['5.12e2',512],['0e-9999',0],['-0.000',0]]as const)assert.deepEqual(parse(text,true),{ok:true,value},text);
 for(const [kind,name,bad]of [[kinds[1],'spinCount',2],[kinds[1],'spinCount',129],[kinds[2],'sampleCount',0],[kinds[2],'sampleCount',513]]as const){const input=preset(kind,0);assert.equal(isCorrelationInput({...input,parameters:{...input.parameters,[name]:bad}}),false);}
});
