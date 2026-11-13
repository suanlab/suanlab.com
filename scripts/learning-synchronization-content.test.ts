import { createRequire } from 'node:module';
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
  const source=readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`,'utf8'),draft=readFileSync(`docs/online-learning/physics-math-for-ai/${kind}-chapter.${lang}.md`,'utf8');let original=source;
  if(kind===kinds[0]){
   const start=source.search(/^## (Research extension:|연구 확장:)/m),end=source.indexOf('\n## ',start+4)+1;
   assert(start>=0&&end>start);original=source.slice(0,start)+source.slice(end);
   const correction={"en": ["Reader, Worker and visualization registration are still pending, as are the pairwise and collective families.", "The reader, Worker, visualization and the pairwise and collective families are now registered; their integration evidence is linked in the research progress record."], "ko": ["아직 읽기 화면·Worker·시각화 등록을 하지 않았으며 두 진동자와 다수 진동자의 가족도 남아 있다.", "현재는 읽기 화면·Worker·시각화와 두 진동자·다수 진동자 가족이 모두 등록되어 있으며 통합 검증 근거는 연구 진행표에 연결되어 있다."]} as Record<string,readonly string[]>;
   original=original.replace(correction[lang][1],correction[lang][0]);
  }
  if(kind!==kinds[0]){
   const start=source.search(/^## (Research extension:|연구 확장:)/m),end=source.indexOf('\n## ',start+4)+1;
   assert(start>=0&&end>start);original=source.slice(0,start)+source.slice(end);
   const corrections={"two-oscillator-locking-phase-slips": {"en": ["The present numerical foundation and draft still await Worker, teaching UI and reader registration.", "The numerical foundation, Worker, teaching UI and reader are now registered; the research progress record links their integration evidence."], "ko": ["현재 원고·수치 기반은 Worker·교수 화면·읽기 등록 전 단계다.", "현재 원고·수치 기반과 Worker·교수 화면·reader가 등록되어 있으며 연구 진행표에 통합 검증 근거를 연결했다."]}} as Record<string,Record<string,readonly string[]>>;
   const correction=corrections[kind]?.[lang];
   if(correction)original=original.replace(correction[1],correction[0]);
  }
  assert.equal(original,draft.slice(draft.indexOf('## 1.')));assert.equal((source.match(/^## /gm)||[]).length,15);assert.equal((source.match(/^\*\*\d\./gm)||[]).length,6);assert(!source.startsWith('# '));
 }
 const html=await getLearningContent(kind);assert.equal(html.outline.ko.length,15);assert.equal(html.outline.en.length,15);assert(!/[가-힣]/.test(html.en));assert(!html.ko.includes('katex-error'));assert(!html.en.includes('katex-error'));assert.equal((html.en.match(/katex-display/g)||[]).length,18);if(kind===kinds[0])for(const text of ['<td>1/|μ|</td>','<td>1/(2|μ|)</td>']){assert(html.ko.includes(text));assert(html.en.includes(text));}
});
test('registered lab and fixed comparisons render on the server without creating a Worker',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{LanguageProvider}=createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider'),{default:Lab}=await import('../src/components/learning/synchronization-lab'),{default:Concept}=await import('../src/components/learning/synchronization-concept');
 const prior=(globalThis as any).Worker;(globalThis as any).Worker=class{constructor(){throw Error('SSR must not create Worker');}};
 try{for(const kind of kinds){const html=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Lab,{kind}),React.createElement(Concept,{kind,language:'ko'})));assert(html.includes('data-synchronization-lab'));assert(html.includes('data-synchronization-concept'));assert(!html.includes('data-synchronization-result='));assert(!html.includes('NaN'));}}finally{if(prior===undefined)delete(globalThis as any).Worker;else(globalThis as any).Worker=prior;}
});
