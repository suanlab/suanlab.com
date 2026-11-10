import {createRequire} from 'node:module';
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {getLearningContent} from '../src/lib/learning-content';
import {unitaryKinds as kinds} from '../src/lib/learning-unitary';
for(const [family,kind] of kinds.entries())test(`registered unitary reader has full bilingual theory and connected teaching: ${kind}`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert(unit);assert.equal(unit.questions.length,3);assert(unit.prerequisiteIds.every(id=>learningUnits.some(u=>u.id===id)));assert.equal(learningGuides[kind].experiments.length,3);assert.equal(learningGuides[kind].steps.length,3);assert(curriculumDomains.find(d=>d.id==='quantum-physics')!.unitIds.includes(kind));
 for(const lang of ['ko','en']){
  const source=readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),draft=readFileSync(`docs/online-learning/physics-math-for-ai/${kind}-chapter.${lang}.md`,'utf8');assert.equal(source,draft.slice(draft.indexOf('## 1.')));assert.equal((source.match(/^## /gm)||[]).length,12);assert.equal((source.match(/^\*\*\d\./gm)||[]).length,6);assert(!source.startsWith('# '));
 }
 const html=await getLearningContent(kind);assert.equal(html.outline.ko.length,12);assert.equal(html.outline.en.length,12);assert(!/[가-힣]/.test(html.en));assert(!html.ko.includes('katex-error'));assert(!html.en.includes('katex-error'));assert.equal((html.en.match(/katex-display/g)||[]).length,7+family);
});
test('registered lab and fixed comparisons render on the server without creating a Worker',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};
 const React=await import('react');(globalThis as typeof globalThis&{React?:typeof React}).React=React;
 const{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=await import('../src/components/language-provider'),{default:Lab}=await import('../src/components/learning/unitary-lab'),{default:Concept}=await import('../src/components/learning/unitary-concept');
 const prior=(globalThis as any).Worker;(globalThis as any).Worker=class{constructor(){throw Error('SSR must not create Worker');}};
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Lab,{kind}),React.createElement(Concept,{kind,language:'ko'})));assert(html.includes('data-unitary-lab'));assert(html.includes('data-unitary-concept'));assert(!html.includes('data-unitary-result='));assert(!html.includes('NaN'));}}finally{if(prior===undefined)delete(globalThis as any).Worker;else(globalThis as any).Worker=prior;}
});

import {createHash}from'node:crypto';
import prior from './fixtures/pre-unitary-registry-hashes.json';
const digest=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('previous554 units, guides and curriculum retain their content and order as unitary readers are added',()=>{
 for(const u of learningUnits.filter(u=>Object.hasOwn(prior.units,u.id))){assert.equal(digest(u),prior.units[u.id as keyof typeof prior.units],u.id);assert.equal(digest(learningGuides[u.id]),prior.guides[u.id as keyof typeof prior.guides],u.id);}
 assert(learningUnits.length>=Object.keys(prior.units).length+2);assert.deepEqual(learningUnits.filter(u=>Object.hasOwn(prior.units,u.id)).map(u=>u.id),Object.keys(prior.units));assert.equal(new Set(learningUnits.map(u=>u.id)).size,learningUnits.length);
 for(const d of curriculumDomains){const restored={...d,unitIds:d.unitIds.filter(id=>Object.hasOwn(prior.units,id))};assert.equal(digest(restored),prior.domains[d.id as keyof typeof prior.domains],d.id);}
});

test('the full registered reader and cover select dedicated Unitary teaching without SSR side effects',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=await import('../src/components/language-provider'),{InteractiveBookReader}=await import('../src/components/learning/interactive-book'),{default:BookArt}=await import('../src/components/learning/book-art');
 for(const kind of kinds){const unit=learningUnits.find(u=>u.id===kind)!;const html=await getLearningContent(kind);const rendered=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(InteractiveBookReader,{unit,html})));assert(rendered.includes(`data-unitary-lab="${kind}"`));assert(rendered.includes(`data-unitary-concept="${kind}"`));assert(rendered.includes('id="check"'));assert.match(rendered,/href="\/book\/interactive\/?"/);assert(!rendered.includes('<canvas'));assert(renderToStaticMarkup(React.createElement(BookArt,{kind})).includes(`data-unitary-cover="${kind}"`));}
});
