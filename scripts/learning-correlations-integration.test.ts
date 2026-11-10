import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {correlationKinds as kinds,correlationGuides as guides} from '../src/lib/learning-correlations';
import {correlationConceptSteps,correlationConceptQuestions,correlationConceptIntuition} from '../src/lib/learning-correlations-concepts';
import {getLearningContent} from '../src/lib/learning-content';
import {LanguageProvider} from '../src/components/language-provider';
import BookArt from '../src/components/learning/book-art';
import {createRequire} from 'node:module';
createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};
(globalThis as typeof globalThis&{React?:typeof React}).React=React;
const baseline=JSON.parse(readFileSync('scripts/fixtures/pre-correlations-registry-hashes.json','utf8'));
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all 524 previous units and guides retain values and order; three new readers and nine guides follow',()=>{
 assert.ok(learningUnits.length>=527);assert.equal(baseline.ids.length,524);assert.deepEqual(learningUnits.slice(0,524).map(u=>u.id),baseline.ids);assert.deepEqual(Object.keys(learningGuides).slice(0,524),baseline.guideOrder);
 for(const unit of learningUnits.slice(0,524)){assert.equal(hash(unit),baseline.unitHashes[unit.id],unit.id);assert.equal(hash(learningGuides[unit.id]),baseline.guideHashes[unit.id],unit.id);}
 assert.deepEqual(learningUnits.slice(524,527).map(u=>u.id),kinds);assert.equal(learningUnits.slice(0,527).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1581);
});
test('historical thermodynamics additions and all prior domain metadata/member prefixes persist',()=>{
 for(const {unitIds,...domain} of curriculumDomains){assert.equal(hash(domain),baseline.curriculumDomainHashesWithoutUnitIds[domain.id],domain.id);const expected=domain.id==='thermo-statmech'?[...baseline.curriculumDomainUnitIds[domain.id],...kinds]:baseline.curriculumDomainUnitIds[domain.id];assert.deepEqual(unitIds.slice(0,expected.length),expected);}
});
for(const kind of kinds)test(`${kind}: actual reader renders its complete bilingual manuscript and dedicated teaching components without a worker during SSR`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert.ok(unit);assert.ok(unit.prerequisiteIds.every(id=>{const i=learningUnits.findIndex(u=>u.id===id);return i>=0&&i<learningUnits.indexOf(unit);}));
 assert.deepEqual(unit.questions,correlationConceptQuestions[kind]);assert.deepEqual(learningGuides[kind].intuition,correlationConceptIntuition[kind]);assert.deepEqual(learningGuides[kind].steps,correlationConceptSteps[kind].map(({title,explanation,formula,diagram})=>({title,explanation,formula,diagram})));assert.deepEqual(learningGuides[kind].experiments,guides[kind].map(({presetIndices,...g})=>({...g,preset:presetIndices[0]})));
 const html=await getLearningContent(kind);for(const lang of ['ko','en'] as const){assert.equal(html.outline[lang].length,[14,15,16][kinds.indexOf(kind)]);assert.doesNotMatch(html[lang],/katex-error|NaN|undefined/);if(lang==='en')assert.doesNotMatch(html[lang],/[가-힣]/);}
 const descriptor=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR worker');}});
 try{const {InteractiveBookReader}=await import('../src/components/learning/interactive-book');const rendered=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(InteractiveBookReader,{unit,html})));assert.ok(rendered.includes(`data-correlation-lab="${kind}"`));assert.ok(rendered.includes(`data-correlation-concept="${kind}"`));const cover=renderToStaticMarkup(React.createElement(BookArt,{kind}));assert.ok(cover.includes(`data-correlation-cover="${kind}"`));assert.match(rendered,/href="\/book\/interactive\/?"/);assert.equal(reads,0);}finally{if(descriptor)Object.defineProperty(globalThis,'Worker',descriptor);else Reflect.deleteProperty(globalThis,'Worker');}
});
