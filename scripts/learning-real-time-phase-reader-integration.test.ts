import {assertPreservedLearningFile} from './learning/assert-manuscript-preservation';
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
import {learningUnits,learningGuides} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {realTimePhaseReader as reader} from '../src/data/real-time-phase-reader';
import {realTimePhaseTeachingId as id} from '../src/lib/learning-real-time-phase-teaching';
import {getLearningContent} from '../src/lib/learning-content';
const before=JSON.parse(readFileSync('scripts/fixtures/pre-real-time-phase-registry-hashes.json','utf8')) as {unitOrder:string[];guideOrder:string[];units:Record<string,string>;guides:Record<string,string>;domains:Record<string,string>;manuscripts:Record<string,string>;preservedFiles:Record<string,string>};
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex'),root='.runtime/learning/real-time-phase-reader-integration';mkdirSync(root,{recursive:true});
test('phase unit appends after600 with prerequisites,13concepts,6guides and8questions',async()=>{
 const unit=learningUnits.find(u=>u.id===id);assert(unit);assert.equal(learningUnits.indexOf(unit),600);assert.deepEqual(unit,reader.unit);assert.deepEqual(learningGuides[id],reader.guide);assert.equal(unit.questions.length,8);assert.equal(reader.guide.steps.length,13);assert.equal(reader.guide.experiments.length,6);
 assert(unit.prerequisiteIds.every(prior=>learningUnits.findIndex(u=>u.id===prior)>=0&&learningUnits.findIndex(u=>u.id===prior)<600));assert(curriculumDomains.find(d=>d.id==='field-multiscale')!.unitIds.includes(id));
 const content=await getLearningContent(id);for(const language of ['ko','en']as const){assert.equal(content.outline[language].length,34);assert(!content[language].includes('katex-error'));assert(!/<h1/.test(content[language]));assert(content[language].includes('<math'));}assert(!/[가-힣]/.test(content.en));
});
test('earlier600 units/guides/scopes/menu andmanuscript substance survive declared wording corrections',()=>{
 assert.equal(before.unitOrder.length,600);assert(learningUnits.length>=601);assert.equal(new Set(learningUnits.map(u=>u.id)).size,learningUnits.length);assert.deepEqual(learningUnits.slice(0,600).map(u=>u.id),before.unitOrder);assert.deepEqual(Object.keys(learningGuides).slice(0,600),before.guideOrder);
 for(const unit of learningUnits.slice(0,600)){assert.equal(hash(unit),before.units[unit.id]);assert.equal(hash(learningGuides[unit.id]),before.guides[unit.id]);}
 for(const[p,h]of Object.entries({...before.manuscripts,...before.preservedFiles}))assertPreservedLearningFile(p,h);
 assert.equal(learningUnits.slice(0,601).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1863);for(const domain of curriculumDomains)assert.equal(hash({...domain,unitIds:domain.unitIds.filter(prior=>before.unitOrder.includes(prior))}),before.domains[domain.id]);
});
test('reader dispatches phase cover/concepts/lazy lab and excludes duplicate embedded questions',()=>{
 const read=(p:string)=>readFileSync(`src/components/learning/${p}`,'utf8');assert(read('interactive-book.tsx').includes('isRealTimePhaseTeachingId(unit.id) ? <RealTimePhaseReaderLab'));assert(read('concept-walkthrough.tsx').includes('isRealTimePhaseTeachingId(kind)) return <RealTimePhaseConcept'));assert(read('book-art.tsx').includes('isRealTimePhaseTeachingId(kind)) return <RealTimePhaseCover'));assert(read('real-time-phase-reader-lab.tsx').includes('id="experiment"'));assert(read('real-time-phase-reader-lab.tsx').includes('language={language} embedded'));assert(read('real-time-phase-lab.tsx').includes('questionsVisible={!embedded}'));
});
test('cover,13reference drawings andembedded SSR render bilingual content without Worker access',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),savedReact=Object.getOwnPropertyDescriptor(globalThis,'React'),savedWorker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{const{default:Graphic}=await import('../src/components/learning/real-time-phase-graphic'),{default:Concept}=await import('../src/components/learning/real-time-phase-concept'),{default:Lab}=await import('../src/components/learning/real-time-phase-lab'),{default:Art}=await import('../src/components/learning/book-art'),{LanguageProvider}=createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider');
  for(const language of ['ko','en']as const){for(let step=0;step<13;step++){const svg=renderToStaticMarkup(React.createElement(Graphic,{step,language}));assert(svg.includes('role="img"')&&svg.includes('<desc'));assert(!svg.includes('NaN'));if(language==='en')assert(!/[가-힣]/.test(svg));}assert(renderToStaticMarkup(React.createElement(Concept,{language})).includes('href="#experiment"'));const embedded=renderToStaticMarkup(React.createElement(Lab,{language,embedded:true}));assert(!embedded.includes('data-real-time-question='));assert(embedded.includes('data-real-time-lessons'));if(language==='en')assert(!/[가-힣]/.test(embedded));}
  const cover=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Art,{kind:id})));assert(cover.includes('role="img"'));assert(!cover.includes('min-width:720'));assert.equal(calls,0);
 }finally{if(savedReact)Object.defineProperty(globalThis,'React',savedReact);else Reflect.deleteProperty(globalThis,'React');if(savedWorker)Object.defineProperty(globalThis,'Worker',savedWorker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('full reader excludes phase numerical engines andindependent mathematical checkers',async()=>{
 const result=await build({entryPoints:['src/components/learning/interactive-book.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:root+'/graph',metafile:true,jsx:'automatic',logLevel:'silent'}),inputs=Object.keys(result.metafile!.inputs);
 assert(!inputs.some(p=>/learning-(?:real-time-phase|saddle-fluctuation|euclidean-kernel|brownian-bridge|spatial-spectrum|nongaussian-field|gaussian-field|duffing-secular|boundary-layer|quartic-asymptotic|regular-perturbation|graph-gauss|gauge-orbits|field-modes|field-balance|scalar-field|quantum-hilbert-core)(?:\.ts|-exact\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)),inputs.join('\n'));const allowed=new Set(['teaching','input','protocol','lesson-state','matrix-view','state','controller','worker-factory','reference','wave-view'].map(name=>'src/lib/learning-real-time-phase-'+name+'.ts'));assert(inputs.filter(p=>p.startsWith('src/lib/learning-real-time-')).every(p=>allowed.has(p)));assert(inputs.includes('src/components/learning/real-time-phase-matrix-engine.ts'));writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(result.metafile,null,2)+'\n');
});
test('all registered bilingual manuscripts retain strict math andEnglish-only visible copy',async()=>{
 const rows:{id:string;koSections:number;enSections:number}[]=[];
 for(let start=0;start<learningUnits.length;start+=6)await Promise.all(learningUnits.slice(start,start+6).map(async unit=>{const content=await getLearningContent(unit.id);for(const language of ['ko','en']as const){assert(content.outline[language].length>0,unit.id);assert(!content[language].includes('katex-error'),unit.id);assert(!/<h1(?:\s|>)/.test(content[language]),unit.id);}assert(!/[가-힣]/.test(content.en),unit.id);rows.push({id:unit.id,koSections:content.outline.ko.length,enSections:content.outline.en.length});}));assert.equal(rows.length,learningUnits.length);writeFileSync(root+'/all-manuscripts.json',JSON.stringify({registeredUnits:learningUnits.length,bilingualManuscripts:rows.length*2,rows},null,2)+'\n');
});
