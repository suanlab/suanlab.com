import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {readFileSync} from 'node:fs';
import {phaseKinds as kinds,phasePresets as presets,phaseGuides as guides,phasePresetInput as preset,parsePhaseNumber as parse} from '../src/lib/learning-phase-transitions';
import {phaseConceptSteps as steps,phaseConceptInputs as inputs,phaseConceptQuestions as questions} from '../src/lib/learning-phase-transitions-concepts';
import {isPhaseInput,phaseInputKey as key,type PhaseInput} from '../src/lib/learning-phase-transitions-protocol';
import {evaluatePhaseRequest as evaluate} from '../src/lib/learning-phase-transitions-evaluator';
import {PhaseResultView,phaseDrawingValue,phaseDrawingResolved,phaseQuantity,phaseMagnetization} from '../src/components/learning/phase-transitions-view';
import PhaseLab from '../src/components/learning/phase-transitions-lab';
import PhaseConceptPanel from '../src/components/learning/phase-transitions-concept';
import {usePhaseSeries} from '../src/components/learning/use-phase-transitions-series';
import {LanguageProvider} from '../src/components/language-provider';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
(globalThis as typeof globalThis&{React?:typeof React}).React=React;
const cache=new Map<string,ReturnType<typeof evaluate>>();
function result(input:PhaseInput){const k=key(input);if(!cache.has(k))cache.set(k,evaluate({version:1,requestId:1,input}));const r=cache.get(k)!;assert.ok(r.ok);return r;}
for(const kind of kinds)test(`${kind}: substantive guides cover every distinct preset and fixed steps remain owned and independent`,()=>{
 assert.equal(guides[kind].length,3);assert.equal(steps[kind].length,6);assert.equal(questions[kind].length,3);
 assert.deepEqual([...new Set(guides[kind].flatMap(g=>g.presetIndices))].sort((a,b)=>a-b),presets[kind].map((_,i)=>i));assert.equal(new Set(presets[kind].map(x=>x.id)).size,presets[kind].length);
 for(const g of guides[kind])for(const lang of ['ko','en'] as const){for(const field of [g.title,g.predict,g.action,g.explain]){assert.ok(field[lang].trim().length>0);if(lang==='en')assert.doesNotMatch(field[lang],/[가-힣]/);}assert.ok(g.explain[lang].length>g.predict[lang].length);}
 for(const [i,s] of steps[kind].entries()){
  assert.equal(s.check.options.length,3);assert.ok(s.check.answer>=0&&s.check.answer<3);assert.ok(s.samples.length>=1);assert.ok(s.explanation.en.length>150);assert.ok(inputs(kind,i).every(isPhaseInput));
  const a=inputs(kind,i),b=inputs(kind,i);assert.notEqual(a[0],b[0]);assert.ok(Object.isFrozen(a[0].parameters));
 }
});
for(const kind of kinds)for(const [i,p] of presets[kind].entries())test(`${p.id}: actual experiment result renders bilingual scientific status without losing root roles`,()=>{
 const r=result(preset(kind,i));for(const language of ['ko','en'] as const){const html=renderToStaticMarkup(React.createElement(PhaseResultView,{response:r,language}));assert.ok(html.includes(`data-phase-result="${kind}"`));assert.ok(html.includes('<table'));assert.doesNotMatch(html,/NaN|undefined|katex-error/);if(language==='en')assert.doesNotMatch(html,/[가-힣]/);}
});
for(const kind of kinds)test(`${kind}: every fixed concept preparation computes its declared model and visual result`,()=>{
 for(let step=0;step<6;step++)for(const input of inputs(kind,step)){const r=result(input);const html=renderToStaticMarkup(React.createElement(PhaseResultView,{response:r,language:'en'}));assert.ok(html.includes(`data-phase-result="${input.kind}"`));assert.doesNotMatch(html,/[가-힣]|NaN|undefined/);}
});
test('scientific preparation checks distinguish modes, symmetric averages, exact criticality, folds and unresolved selection',()=>{
 const f=result(preset(kinds[0],1));assert.ok(f.kind===kinds[0]);assert.deepEqual(f.result.modes.candidateTotalMagnetizations,[0]);assert.ok(f.result.states.every(x=>x.probability.value===1/3));
 const symmetric=result(preset(kinds[0],4));assert.ok(symmetric.kind===kinds[0]);assert.equal(symmetric.result.meanMagnetization.resolution,'structural-zero');
 assert.deepEqual([3,4,5].map(i=>{const r=result(preset(kinds[1],i));assert.ok(r.kind===kinds[1]);return r.result.rootCount;}),[3,1,1]);
 const c=result(preset(kinds[1],1));assert.ok(c.kind===kinds[1]);assert.equal(c.result.roots[0].branchFieldDerivative.resolution,'mathematical-infinity');
 const l=result(preset(kinds[2],5));assert.ok(l.kind===kinds[2]);assert.deepEqual(l.result.globalMinimumIndices,[0,2,4]);
 const unresolved=result(preset(kinds[2],11));assert.ok(unresolved.kind===kinds[2]);assert.equal(unresolved.result.rootsComplete,true);assert.equal(unresolved.result.globalSelectionResolved,false);assert.equal(unresolved.result.globalMinimumCandidates.length,2);
});
test('drawing functions are explicitly separate from root certification and respect each domain',()=>{
 const c=result(preset(kinds[1],2)),l=result(preset(kinds[2],5));assert.ok(c.kind===kinds[1]&&l.kind===kinds[2]);assert.equal(phaseDrawingValue(c,1,'derivative'),null);assert.equal(phaseDrawingValue(c,2,'potential'),null);
 assert.ok(Number.isFinite(phaseDrawingValue(l,2,'potential')));assert.equal(phaseDrawingValue(l,0,'potential'),0);assert.equal(phaseDrawingValue(l,.5,'potential'),1/96);
});
test('decimal controls preserve zero boundaries and reject nonzero underflow and malformed text',()=>{
 for(const text of ['1e-9999','NaN','Infinity','1,2','','0x10'])assert.equal(parse(text).ok,false,text);
 assert.deepEqual(parse('5e-324'),{ok:true,value:Number.MIN_VALUE});assert.deepEqual(parse('-0'),{ok:true,value:0});assert.deepEqual(parse('0e-9999'),{ok:true,value:0});
 assert.throws(()=>preset(kinds[0],.5),RangeError);assert.throws(()=>inputs(kinds[1],6),RangeError);
});
test('SSR experiments, concept series and panels never access Worker or execute numerical work',()=>{
 const original=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR worker');}});
 function Probe(){const r=usePhaseSeries([preset(kinds[0],0)]);assert.equal(r.status,'pending');assert.deepEqual(r.responses,[]);return React.createElement('output',null,'pending');}
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(PhaseLab,{kind}),React.createElement(PhaseConceptPanel,{kind,language:'en'})));assert.ok(html.includes('data-phase-ready="false"'));assert.ok(html.includes('data-phase-concept-state="pending"'));assert.ok(!html.includes('data-phase-result='));}renderToStaticMarkup(React.createElement(Probe));assert.equal(reads,0);}finally{if(original)Object.defineProperty(globalThis,'Worker',original);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('registered phase teaching stays aligned with the three dedicated modules',()=>{
 assert.ok(learningUnits.length>=524);assert.equal(Object.values(learningGuides).slice(0,524).reduce((n,g)=>n+g.experiments.length,0),1572);const ids:string[]=learningUnits.map(u=>u.id);for(const kind of kinds)assert.ok(ids.includes(kind));
 const config=readFileSync('src/lib/learning-phase-transitions.ts','utf8');assert.doesNotMatch(config,/import \{[^}]+\} from '.\/learning-(?:curie-weiss|landau-sixth-order)'/);
});

test('missing computed values and a fold derivative do not claim that the limiting pressure is undefined',()=>{
 assert.equal(phaseQuantity(null,(_ko,en)=>en),'Not available');assert.equal(phaseQuantity(null,(ko)=>ko),'미제공');
 const input={...preset(kinds[1],3),options:{maximumPrecision:128}} as PhaseInput,r=result(input);assert.ok(r.kind===kinds[1]);assert.equal(r.result.fullyResolved,false);const html=renderToStaticMarkup(React.createElement(PhaseResultView,{response:r,language:'en'}));assert.ok(html.includes('Not available'));assert.doesNotMatch(html,/Undefined/);
 const fold=result(preset(kinds[2],4));assert.ok(fold.kind===kinds[2]);const view=renderToStaticMarkup(React.createElement(PhaseResultView,{response:fold,language:'en'}));assert.ok(view.includes('No single-valued derivative'));
});

test('a close spinodal pair keeps certified roots while floating drawing noise is labeled unresolved',()=>{
 const r=result(preset(kinds[1],3));assert.ok(r.kind===kinds[1]);assert.equal(r.result.rootCount,3);const center=r.result.roots[0].magnetization.value!,radius=1e-8;
 const samples=Array.from({length:321},(_,i)=>{const x=center-radius+2*radius*i/320;return {x,y:phaseDrawingValue(r,x,'derivative')!};});assert.equal(phaseDrawingResolved(r,'derivative',samples),false);assert.notEqual(phaseMagnetization(r.result.roots[0].magnetization,(_ko,en)=>en),phaseMagnetization(r.result.roots[1].magnetization,(_ko,en)=>en));const html=renderToStaticMarkup(React.createElement(PhaseResultView,{response:r,language:'en'}));assert.ok(html.includes(String(r.result.roots[0].magnetization.value)));assert.ok(html.includes(String(r.result.roots[1].magnetization.value)));
});


test('spin-count decimal lexemes reject fractional digits lost by binary64 rounding',()=>{
 for(const text of ['3.00000000000000001','2.99999999999999999','1.2800000000000000001e2','127.99999999999999999','1e-9999'])assert.deepEqual(parse(text,true),{ok:false},text);
 assert.deepEqual(parse('3.00000000000000001'),{ok:true,value:3});
});
test('spin-count decimal lexemes accept exact integer decimal and exponent notation',()=>{
 for(const [text,value]of [['3.000',3],['30e-1',3],['.03e2',3],['1.28e2',128],['0e-9999',0],['-0.000',0]]as const)assert.deepEqual(parse(text,true),{ok:true,value},text);
 // Parsing a number is separate from the model domain: N still must lie in1..128.
 for(const spinCount of [0,129]){const input=preset(kinds[0],0);assert.equal(isPhaseInput({...input,parameters:{...input.parameters,spinCount}}),false);}
});
