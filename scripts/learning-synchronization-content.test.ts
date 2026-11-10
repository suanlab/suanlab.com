import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {getLearningContent} from '../src/lib/learning-content';
import {synchronizationKinds as kinds} from '../src/lib/learning-synchronization';
for(const kind of kinds)test(`registered synchronization reader has full bilingual theory and connected teaching: ${kind}`,async()=>{
 const unit=learningUnits.find(u=>u.id===kind)!;assert(unit);assert.equal(unit.questions.length,3);assert(unit.prerequisiteIds.every(id=>learningUnits.some(u=>u.id===id)));assert.equal(learningGuides[kind].experiments.length,3);assert.equal(learningGuides[kind].steps.length,3);assert(curriculumDomains.find(d=>d.id==='nonequilibrium')!.unitIds.includes(kind));
 for(const lang of ['ko','en']){
  const source=readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),draft=readFileSync(`docs/online-learning/physics-math-for-ai/${kind}-chapter.${lang}.md`,'utf8');assert.equal(source,draft.slice(draft.indexOf('## 1.')));assert.equal((source.match(/^## /gm)||[]).length,14);assert.equal((source.match(/^\*\*\d\./gm)||[]).length,6);assert(!source.startsWith('# '));
 }
 const html=await getLearningContent(kind);assert.equal(html.outline.ko.length,14);assert.equal(html.outline.en.length,14);assert(!/[가-힣]/.test(html.en));assert(!html.ko.includes('katex-error'));assert(!html.en.includes('katex-error'));assert.equal((html.en.match(/katex-display/g)||[]).length,10);if(kind===kinds[0])for(const text of ['<td>1/|μ|</td>','<td>1/(2|μ|)</td>']){assert(html.ko.includes(text));assert(html.en.includes(text));}
});
test('registered lab and fixed comparisons render on the server without creating a Worker',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=await import('../src/components/language-provider'),{default:Lab}=await import('../src/components/learning/synchronization-lab'),{default:Concept}=await import('../src/components/learning/synchronization-concept');
 const prior=(globalThis as any).Worker;(globalThis as any).Worker=class{constructor(){throw Error('SSR must not create Worker');}};
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Lab,{kind}),React.createElement(Concept,{kind,language:'ko'})));assert(html.includes('data-synchronization-lab'));assert(html.includes('data-synchronization-concept'));assert(!html.includes('data-synchronization-result='));assert(!html.includes('NaN'));}}finally{if(prior===undefined)delete(globalThis as any).Worker;else(globalThis as any).Worker=prior;}
});
