import {createRequire} from 'node:module';
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {getLearningContent} from '../src/lib/learning-content';
import {densityKinds as kinds} from '../src/lib/learning-density';
for(const [family,kind] of kinds.entries())test(`registered density reader has full bilingual theory and connected teaching: ${kind}`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert(unit);assert.equal(unit.questions.length,3);assert(unit.prerequisiteIds.every(id=>learningUnits.some(u=>u.id===id)));assert.equal(learningGuides[kind].experiments.length,3);assert.equal(learningGuides[kind].steps.length,3);assert(curriculumDomains.find(d=>d.id==='quantum-physics')!.unitIds.includes(kind));
 for(const lang of ['ko','en']){
  const source=readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),draft=readFileSync(`docs/online-learning/physics-math-for-ai/${kind}-chapter.${lang}.md`,'utf8');const start=source.search(/^## (Research extension:|연구 확장:)/m),end=source.indexOf('\n## ',start+4)+1;assert(start>=0&&end>start);assert.equal(source.slice(0,start)+source.slice(end),draft.slice(draft.indexOf('## 1.')));assert.equal((source.match(/^## /gm)||[]).length,15);assert.equal((source.match(/^\*\*\d\./gm)||[]).length,8);assert(!source.startsWith('# '));
 }
 const html=await getLearningContent(kind);assert.equal(html.outline.ko.length,15);assert.equal(html.outline.en.length,15);assert(!/[가-힣]/.test(html.en));assert(!html.ko.includes('katex-error'));assert(!html.en.includes('katex-error'));assert.equal((html.en.match(/katex-display/g)||[]).length,20);
});
test('registered lab and fixed comparisons render on the server without creating a Worker',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};
 const React=await import('react');(globalThis as typeof globalThis&{React?:typeof React}).React=React;
 const{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider'),{default:Lab}=await import('../src/components/learning/density-lab'),{default:Concept}=await import('../src/components/learning/density-concept');
 const prior=(globalThis as any).Worker;(globalThis as any).Worker=class{constructor(){throw Error('SSR must not create Worker');}};
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Lab,{kind}),React.createElement(Concept,{kind,language:'ko'})));assert(html.includes('data-density-lab'));assert(html.includes('data-density-concept'));assert(!html.includes('data-density-result='));assert(!html.includes('NaN'));}}finally{if(prior===undefined)delete(globalThis as any).Worker;else(globalThis as any).Worker=prior;}
});

import {createHash}from'node:crypto';
import prior from './fixtures/pre-density-registry-hashes.json';
const digest=(v:unknown)=>createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('previous558 units, guides and curriculum retain their content and order as density readers are added',()=>{
 for(const u of learningUnits.filter(u=>Object.hasOwn(prior.units,u.id))){assert.equal(digest(u),prior.units[u.id as keyof typeof prior.units],u.id);assert.equal(digest(learningGuides[u.id]),prior.guides[u.id as keyof typeof prior.guides],u.id);}
 assert(learningUnits.length>=Object.keys(prior.units).length+2);assert.deepEqual(learningUnits.filter(u=>Object.hasOwn(prior.units,u.id)).map(u=>u.id),Object.keys(prior.units));assert.equal(new Set(learningUnits.map(u=>u.id)).size,learningUnits.length);
 for(const d of curriculumDomains){const restored={...d,unitIds:d.unitIds.filter(id=>Object.hasOwn(prior.units,id))};assert.equal(digest(restored),prior.domains[d.id as keyof typeof prior.domains],d.id);}
});

test('the full registered reader and cover select dedicated Density teaching without SSR side effects',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider'),{InteractiveBookReader}=await import('../src/components/learning/interactive-book'),{default:BookArt}=await import('../src/components/learning/book-art');
 for(const kind of kinds){const unit=learningUnits.find(u=>u.id===kind)!;const html=await getLearningContent(kind);const rendered=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(InteractiveBookReader,{unit,html})));assert(rendered.includes(`data-density-lab="${kind}"`));assert(rendered.includes(`data-density-concept="${kind}"`));assert(rendered.includes('id="check"'));assert.match(rendered,/href="\/book\/interactive\/?"/);assert(!rendered.includes('<canvas'));assert(renderToStaticMarkup(React.createElement(BookArt,{kind})).includes(`data-density-cover="${kind}"`));}
});
