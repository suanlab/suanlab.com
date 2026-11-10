import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync}from'node:fs';
import {createHash}from'node:crypto';
import React from 'react';
import {renderToStaticMarkup}from'react-dom/server';
import {learningUnits,learningGuides}from'../src/data/interactive-book';
import {curriculumDomains}from'../src/data/learning-curriculum';
import {finiteSizeKinds as kinds,finiteSizeGuides as guides}from'../src/lib/learning-finite-size';
import {finiteSizeConceptSteps as steps,finiteSizeConceptQuestions as questions,finiteSizeConceptIntuition as intuition}from'../src/lib/learning-finite-size-concepts';
import {getLearningContent}from'../src/lib/learning-content';
import {LanguageProvider}from'../src/components/language-provider';
import BookArt from'../src/components/learning/book-art';
import {createRequire}from'node:module';
createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};
(globalThis as typeof globalThis&{React?:typeof React}).React=React;
const baseline=JSON.parse(readFileSync('scripts/fixtures/pre-finite-size-registry-hashes.json','utf8'));
const hash=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('530 previous readers and guides retain values and order while exactly three readers and nine guides are added',()=>{
 assert.ok(learningUnits.length>=533);assert.equal(baseline.ids.length,530);assert.deepEqual(learningUnits.slice(0,530).map(u=>u.id),baseline.ids);assert.deepEqual(Object.keys(learningGuides).slice(0,530),baseline.guideOrder);
 for(const u of learningUnits.slice(0,530)){assert.equal(hash(u),baseline.unitHashes[u.id],u.id);assert.equal(hash(learningGuides[u.id]),baseline.guideHashes[u.id],u.id);}
 assert.deepEqual(learningUnits.slice(530,533).map(u=>u.id),kinds);assert.equal(learningUnits.slice(0,533).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1599);
});
test('historical thermodynamics membership and previous domain metadata and membership order remain intact',()=>{
 for(const {unitIds,...domain}of curriculumDomains){assert.equal(hash(domain),baseline.curriculumDomainHashesWithoutUnitIds[domain.id],domain.id);const prior=domain.id==='thermo-statmech'?[...baseline.curriculumDomainUnitIds[domain.id],...kinds]:baseline.curriculumDomainUnitIds[domain.id];assert.deepEqual(unitIds.slice(0,prior.length),prior);}
});
for(const [index,kind]of kinds.entries())test(`${kind}: actual reader dispatches full bilingual content, cover, concept, lab and assessment without SSR Worker reads`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert.ok(unit);assert.ok(unit.prerequisiteIds.every(id=>learningUnits.findIndex(u=>u.id===id)>=0&&learningUnits.findIndex(u=>u.id===id)<learningUnits.indexOf(unit)));assert.deepEqual(unit.questions,questions[kind]);assert.deepEqual(learningGuides[kind].intuition,intuition[kind]);assert.deepEqual(learningGuides[kind].steps,steps[kind].map(({title,explanation,formula,diagram})=>({title,explanation,formula,diagram})));assert.deepEqual(learningGuides[kind].experiments,guides[kind].map(({presetIndices,...g})=>({...g,preset:presetIndices[0]})));
 const html=await getLearningContent(kind);for(const language of ['ko','en']as const){assert.equal(html.outline[language].length,[17,16,16][index]);assert.doesNotMatch(html[language],/katex-error|NaN|\$\$/);if(language==='en')assert.doesNotMatch(html[language],/[가-힣]/);}
 const descriptor=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{const {InteractiveBookReader}=await import('../src/components/learning/interactive-book');const rendered=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(InteractiveBookReader,{unit,html})));assert.ok(rendered.includes(`data-finite-size-lab="${kind}"`));assert.ok(rendered.includes(`data-finite-size-concept="${kind}"`));assert.ok(rendered.includes('id="check"'));assert.ok(renderToStaticMarkup(React.createElement(BookArt,{kind})).includes(`data-finite-size-cover="${kind}"`));assert.match(rendered,/href="\/book\/interactive\/?"/);assert.equal(reads,0);}finally{if(descriptor)Object.defineProperty(globalThis,'Worker',descriptor);else Reflect.deleteProperty(globalThis,'Worker');}
});
