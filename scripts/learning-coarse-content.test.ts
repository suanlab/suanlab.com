import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {getLearningContent} from '../src/lib/learning-content';
import {coarseKinds as kinds} from '../src/lib/learning-coarse';
for(const kind of kinds)test(`registered coarse reader has full bilingual theory and connected teaching: ${kind}`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert(unit);assert.equal(unit.questions.length,3);assert(unit.prerequisiteIds.every(id=>learningUnits.some(u=>u.id===id)));assert.equal(learningGuides[kind].experiments.length,3);assert.equal(learningGuides[kind].steps.length,3);assert(curriculumDomains.find(d=>d.id==='nonequilibrium')!.unitIds.includes(kind));
 for(const lang of ['ko','en']){
  const source=readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),draft=readFileSync(`docs/online-learning/physics-math-for-ai/${kind}-chapter.${lang}.md`,'utf8');assert.equal(source,draft.slice(draft.indexOf('## 1.')));assert.equal((source.match(/^## /gm)||[]).length,14);assert.equal((source.match(/^\*\*\d\./gm)||[]).length,6);assert(!source.startsWith('# '));
 }
 const html=await getLearningContent(kind);assert.equal(html.outline.ko.length,14);assert.equal(html.outline.en.length,14);assert(!/[가-힣]/.test(html.en));assert(!html.ko.includes('katex-error'));assert(!html.en.includes('katex-error'));assert.equal((html.en.match(/katex-display/g)||[]).length,10);
});
test('registered lab and fixed comparisons render on the server without creating a Worker',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=await import('../src/components/language-provider'),{default:Lab}=await import('../src/components/learning/coarse-lab'),{default:Concept}=await import('../src/components/learning/coarse-concept');
 const prior=(globalThis as any).Worker;(globalThis as any).Worker=class{constructor(){throw Error('SSR must not create Worker');}};
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Lab,{kind}),React.createElement(Concept,{kind,language:'ko'})));assert(html.includes('data-coarse-lab'));assert(html.includes('data-coarse-concept'));assert(!html.includes('data-coarse-result='));assert(!html.includes('NaN'));}}finally{if(prior===undefined)delete(globalThis as any).Worker;else(globalThis as any).Worker=prior;}
});

import {createHash}from'node:crypto';
import prior from './fixtures/pre-coarse-registry-hashes.json';
const digest=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('previous registered units, guides and curriculum retain content and order through later additions',()=>{
 for(const u of learningUnits.filter(u=>Object.hasOwn(prior.units,u.id))){assert.equal(digest(u),prior.units[u.id as keyof typeof prior.units],u.id);assert.equal(digest(learningGuides[u.id]),prior.guides[u.id as keyof typeof prior.guides],u.id);}
 assert(learningUnits.length>=Object.keys(prior.units).length+2);assert.deepEqual(learningUnits.filter(u=>Object.hasOwn(prior.units,u.id)).map(u=>u.id),Object.keys(prior.units));assert.equal(new Set(learningUnits.map(u=>u.id)).size,learningUnits.length);
 for(const d of curriculumDomains){const restored={...d,unitIds:d.unitIds.filter(id=>Object.hasOwn(prior.units,id))};assert.equal(digest(restored),prior.domains[d.id as keyof typeof prior.domains],d.id);}
});
