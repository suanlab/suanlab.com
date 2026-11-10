import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {detailedBalanceKinds as kinds,detailedBalanceGuides as guides} from '../src/lib/learning-detailed-balance';
import {detailedBalanceConceptSteps,detailedBalanceConceptQuestions,detailedBalanceConceptIntuition} from '../src/lib/learning-detailed-balance-concepts';
import {getLearningContent} from '../src/lib/learning-content';
import {LanguageProvider} from '../src/components/language-provider';
import BookArt from '../src/components/learning/book-art';
import {createRequire} from 'node:module';
createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};
(globalThis as typeof globalThis&{React?:typeof React}).React=React;
const baseline=JSON.parse(readFileSync('scripts/fixtures/pre-detailed-balance-registry-hashes.json','utf8'));
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('all 527 previous units and guides retain values and order; three new readers and nine guides follow',()=>{
 assert.ok(learningUnits.length>=530);assert.equal(baseline.ids.length,527);assert.deepEqual(learningUnits.slice(0,527).map(u=>u.id),baseline.ids);assert.deepEqual(Object.keys(learningGuides).slice(0,527),baseline.guideOrder);
 for(const unit of learningUnits.slice(0,527)){assert.equal(hash(unit),baseline.unitHashes[unit.id],unit.id);assert.equal(hash(learningGuides[unit.id]),baseline.guideHashes[unit.id],unit.id);}
 assert.deepEqual(learningUnits.slice(527,530).map(u=>u.id),kinds);assert.equal(learningUnits.slice(0,530).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1590);
});
test('the original detailed-balance curriculum memberships remain unchanged prefixes when later readers are added',()=>{
 for(const {unitIds,...domain} of curriculumDomains){assert.equal(hash(domain),baseline.curriculumDomainHashesWithoutUnitIds[domain.id],domain.id);const expected=domain.id==='thermo-statmech'?[...baseline.curriculumDomainUnitIds[domain.id],...kinds]:baseline.curriculumDomainUnitIds[domain.id];assert.deepEqual(unitIds.slice(0,expected.length),expected);}
});
for(const kind of kinds)test(`${kind}: actual reader renders its complete bilingual manuscript and dedicated teaching components without a worker during SSR`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert.ok(unit);assert.ok(unit.prerequisiteIds.every(id=>{const i=learningUnits.findIndex(u=>u.id===id);return i>=0&&i<learningUnits.indexOf(unit);}));
 assert.deepEqual(unit.questions,detailedBalanceConceptQuestions[kind]);assert.deepEqual(learningGuides[kind].intuition,detailedBalanceConceptIntuition[kind]);assert.deepEqual(learningGuides[kind].steps,detailedBalanceConceptSteps[kind].map(({title,explanation,formula,diagram})=>({title,explanation,formula,diagram})));assert.deepEqual(learningGuides[kind].experiments,guides[kind].map(({presetIndices,...g})=>({...g,preset:presetIndices[0]})));
 const html=await getLearningContent(kind);for(const lang of ['ko','en'] as const){assert.equal(html.outline[lang].length,17);assert.doesNotMatch(html[lang],/katex-error|NaN|undefined/);if(lang==='en')assert.doesNotMatch(html[lang],/[가-힣]/);}
 const descriptor=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR worker');}});
 try{const {InteractiveBookReader}=await import('../src/components/learning/interactive-book');const rendered=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(InteractiveBookReader,{unit,html})));assert.ok(rendered.includes(`data-detailed-balance-lab="${kind}"`));assert.ok(rendered.includes(`data-detailed-balance-concept="${kind}"`));const cover=renderToStaticMarkup(React.createElement(BookArt,{kind}));assert.ok(cover.includes(`data-detailed-balance-cover="${kind}"`));assert.match(rendered,/href="\/book\/interactive\/?"/);assert.equal(reads,0);}finally{if(descriptor)Object.defineProperty(globalThis,'Worker',descriptor);else Reflect.deleteProperty(globalThis,'Worker');}
});
