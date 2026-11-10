import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
import {learningUnits,learningGuides,type LearningUnitId} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {informationTeaching} from '../src/data/information-teaching';
import {informationTeachingIds as ids} from '../src/lib/learning-information-teaching';
import {getLearningContent} from '../src/lib/learning-content';
import before from './fixtures/pre-information-registry-hashes.json';
const runtime='.runtime/learning/quantum-information-reader-integration';mkdirSync(runtime,{recursive:true});
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
for(const[index,id]of ids.entries())test(`information reader registration: ${id}`,async()=>{
 const unit=learningUnits.find(u=>(u.id as string)===id);assert(unit);assert.equal(learningUnits.indexOf(unit),575+index);assert.deepEqual(unit as unknown,informationTeaching[index].unit);assert.deepEqual(learningGuides[unit.id],informationTeaching[index].guide);
 assert.equal(unit.questions.length,[3,5,5,5,5,5,5,6,6][index]);assert.equal(learningGuides[unit.id].experiments.length,index===0?3:4);assert.equal(new Set(unit.questions.map(q=>q.id)).size,unit.questions.length);
 assert(unit.prerequisiteIds.every(prior=>learningUnits.findIndex(u=>u.id===prior)>=0&&learningUnits.findIndex(u=>u.id===prior)<575+index));assert(curriculumDomains.find(d=>d.id==='quantum-physics')!.unitIds.includes(unit.id));
 const content=await getLearningContent(id as LearningUnitId);for(const lang of ['ko','en']as const){assert(content.outline[lang].length>=7);assert(content[lang].includes('katex'));assert(!content[lang].includes('katex-error'));assert(!/<h1/.test(content[lang]));}assert(!/[가-힣]/.test(content.en));
});
test('575 historical units/guides and all51 domain scopes survive exact append-only registration',()=>{
 assert(learningUnits.length>=584);assert.equal(new Set(learningUnits.slice(0,584).map(u=>u.id)).size,584);assert.deepEqual(learningUnits.slice(0,575).map(u=>u.id),before.unitOrder);assert.deepEqual(Object.keys(learningGuides).slice(0,575),before.guideOrder);assert.deepEqual(learningUnits.slice(575,584).map(u=>u.id),[...ids]);assert.deepEqual(Object.keys(learningGuides).slice(575,584),[...ids]);
 for(const unit of learningUnits.slice(0,575)){assert.equal(hash(unit),before.units[unit.id as keyof typeof before.units]);assert.equal(hash(learningGuides[unit.id]),before.guides[unit.id as keyof typeof before.guides]);}
 assert.equal(learningUnits.slice(0,584).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1761);assert.equal(curriculumDomains.length,51);for(const domain of curriculumDomains){const restored={...domain,unitIds:domain.unitIds.filter(id=>(before.unitOrder as string[]).includes(id))};assert.equal(hash(restored),before.domains[domain.id as keyof typeof before.domains],domain.id);}
});
test('the reader connects covers, concepts, experiment anchors and embedded quizzes',()=>{
 const source=(name:string)=>readFileSync(`src/components/learning/${name}`,'utf8');assert(source('interactive-book.tsx').includes('isInformationTeachingId(unit.id) ? <InformationReaderLab'));assert(source('concept-walkthrough.tsx').includes('isInformationTeachingId(kind)) return <InformationConcept'));assert(source('book-art.tsx').includes('isInformationTeachingId(kind)) return <InformationCover'));assert(source('information-reader-lab.tsx').includes('id="experiment"'));assert(source('information-reader-lab.tsx').includes('<Lab language={language} embedded/>'));
 for(const family of ['discrimination','entropy','ensemble','relative','bb84','teleportation','error-correction','majorization','capacity']){const s=source(`information-${family}-lab.tsx`);assert(s.includes('embedded=false'));assert.equal(s.split('{!embedded&&<>').length-1,2);}
});
test('all63 concept references and nine covers render without reading Worker during SSR',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server');const savedReact=Object.getOwnPropertyDescriptor(globalThis,'React'),savedWorker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{const{default:Graphic}=await import('../src/components/learning/information-graphic'),{default:Concept}=await import('../src/components/learning/information-concept'),{default:Art}=await import('../src/components/learning/book-art'),{LanguageProvider}=await import('../src/components/language-provider');let count=0;for(const{unit,guide}of informationTeaching){const cover=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Art,{kind:unit.id})));assert(cover.includes(`data-information-graphic="${unit.id}"`));assert(!cover.includes('<canvas'));for(const lang of ['ko','en']as const){const concept=renderToStaticMarkup(React.createElement(Concept,{kind:unit.id,language:lang}));assert(concept.includes(`data-information-concept="${unit.id}"`));assert(concept.includes('href="#experiment"'));for(let step=0;step<guide.steps.length;step++){const graphic=renderToStaticMarkup(React.createElement(Graphic,{kind:unit.id,step,language:lang}));assert(graphic.includes('<svg'));assert(!graphic.includes('<canvas'));assert(!graphic.includes('NaN'));if(lang==='en')assert(!/[가-힣]/.test(graphic));if(lang==='ko')count++;}}}assert.equal(count,63);assert.equal(reads,0);
 }finally{if(savedReact)Object.defineProperty(globalThis,'React',savedReact);else Reflect.deleteProperty(globalThis,'React');if(savedWorker)Object.defineProperty(globalThis,'Worker',savedWorker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('the complete reader dependency graph keeps all information engines off the main thread',async()=>{
 const r=await build({entryPoints:['src/components/learning/interactive-book.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:`${runtime}/graph`,metafile:true,jsx:'automatic',logLevel:'silent'});const inputs=Object.keys(r.metafile!.inputs);
 assert(!inputs.some(p=>/learning-quantum-information-(?:calculation|response|.*validator|evaluator|entropies|capacity\.ts|majorization\.ts|bb84\.ts|protocols\.ts)/.test(p)));assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts'));writeFileSync(`${runtime}/reader-main-thread-dependency-graph.json`,JSON.stringify({informationNumericalEnginesExcluded:true,inputs,metafile:r.metafile},null,2)+'\n');
});
