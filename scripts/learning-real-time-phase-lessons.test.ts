import test from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {realTimePhaseGuides as guides,realTimePhaseGuidePreparations as preparations,realTimePhaseQuestions as questions,realTimePhaseSteps} from '../src/data/real-time-phase-teaching';
import {emptyRealTimeLesson,readRealTimeLesson} from '../src/lib/learning-real-time-phase-lesson-state';
import {readRealTimePhaseInput,realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
import Lessons,{realTimeLessonLimits as limits} from '../src/components/learning/real-time-phase-lessons';

test('six bilingual guides bind every named preparation and retain all original inputs',()=>{
  assert.equal(guides.length,6);assert.equal(questions.length,8);assert.equal(preparations.length,17);assert.equal(realTimePhaseExamples.length,32);assert.equal(realTimePhaseSteps.length,13);
  assert.equal(new Set(preparations.map(p=>p.id)).size,17);
  const used=new Set(guides.flatMap(g=>[...g.preparations]));assert.deepEqual([...used].sort(),preparations.map(p=>p.id).sort());
  for(const prep of preparations){const copy=readRealTimePhaseInput(prep.input);assert.deepEqual(copy,prep.input,prep.id);assert(Object.isFrozen(copy));}
  const inspect=(v:unknown)=>{if(v&&typeof v==='object'){const d=v as Record<string,unknown>;if(typeof d.en==='string'){assert(d.en.length>0);assert(!/[가-힣]/u.test(d.en),d.en);}Object.values(d).forEach(inspect);}};
  [guides,questions,preparations].forEach(inspect);
  for(const q of questions){assert.equal(q.options.length,3);assert(Number.isInteger(q.correct)&&q.correct>=0&&q.correct<q.options.length);assert(q.explanation.ko&&q.explanation.en);}
});
test('learning state owns arrays, preserves predictions and rejects malformed or accessor data',()=>{
  const empty=emptyRealTimeLesson(limits),owned=readRealTimeLesson(empty,limits)!;assert(owned);assert.notEqual(owned,empty);assert.notEqual(owned.answers,empty.answers);assert(Object.isFrozen(owned)&&Object.isFrozen(owned.answers));
  const valid={...empty,guide:5,explanationOpen:true,predictions:['x\ny','',...Array(4).fill('')],answers:questions.map(q=>q.correct)};assert.deepEqual(readRealTimeLesson(valid,limits),valid);
  for(const patch of [{guide:-1},{guide:6},{guide:1.5},{explanationOpen:'yes'},{answers:[0]},{answers:Array(8).fill(3)},{predictions:Array(6).fill('x'.repeat(2001))},{predictions:Array(6).fill(7)},{extra:1}])assert.equal(readRealTimeLesson({...empty,...patch},limits),null);
  assert.equal(readRealTimeLesson(null,limits),null);let reads=0;const getter={...empty};Object.defineProperty(getter,'guide',{get(){reads++;return 0;}});assert.equal(readRealTimeLesson(getter,limits),null);
  const answers=[...empty.answers];Object.defineProperty(answers,'0',{get(){reads++;return 0;}});assert.equal(readRealTimeLesson({...empty,answers},limits),null);assert.equal(reads,0);
});
test('SSR renders all guides and feedback without calculating or applying inputs',()=>{
  let applied=0;
  for(const language of ['ko','en'] as const)for(let guide=0;guide<guides.length;guide++){
    const state={...emptyRealTimeLesson(limits),guide,explanationOpen:true,answers:questions.map(q=>q.correct)};
    const html=renderToStaticMarkup(createElement(Lessons,{language,state,onApply:()=>{applied++;}}));
    assert(html.includes('data-real-time-guide-explanation'));assert(html.includes(renderToStaticMarkup(createElement('p',null,guides[guide].explanation[language])).slice(3,-4)));assert.equal((html.match(/data-real-time-answer="correct"/g)||[]).length,8);if(language==='en')assert(!/[가-힣]/u.test(html));
  }
  const reset=renderToStaticMarkup(createElement(Lessons,{language:'en',onApply:()=>{applied++;}}));assert(!reset.includes('data-real-time-answer='));assert(!reset.includes('data-real-time-guide-explanation'));
  assert.equal(applied,0);
});
test('lesson-only browser graph contains no calculator, validator, Worker or Three.js',async()=>{
  const output=await build({entryPoints:['src/components/learning/real-time-phase-lessons.tsx'],bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',metafile:true});
  const inputs=Object.keys(output.metafile!.inputs);assert(!inputs.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation|-controller|-evaluator|-worker-factory)|learning-quantum-hilbert-core|three\//.test(p)),inputs.join('\n'));
});
