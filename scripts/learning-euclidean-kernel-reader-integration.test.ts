import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
import {learningUnits,learningGuides,type LearningUnitId} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {euclideanKernelReader} from '../src/data/euclidean-kernel-reader';
import {euclideanKernelTeachingId as id} from '../src/lib/learning-euclidean-kernel-teaching';
import {getLearningContent} from '../src/lib/learning-content';
import before from './fixtures/pre-euclidean-kernel-registry-hashes.json';
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const root='.runtime/learning/euclidean-kernel-reader-integration';mkdirSync(root,{recursive:true});
test('Euclidean-kernel unit appends with valid metadata,prerequisites,eight concepts,six guides and eight questions',async()=>{
 const unit=learningUnits.find(u=>String(u.id)===id);assert(unit);assert.equal(learningUnits.indexOf(unit),598);assert.deepEqual(unit as unknown,euclideanKernelReader.unit);assert.deepEqual(learningGuides[unit.id],euclideanKernelReader.guide);assert.equal(unit.questions.length,8);assert.equal(euclideanKernelReader.guide.steps.length,8);assert.equal(euclideanKernelReader.guide.experiments.length,6);
 assert(unit.prerequisiteIds.every(prior=>learningUnits.findIndex(u=>u.id===prior)>=0&&learningUnits.findIndex(u=>u.id===prior)<598));assert(curriculumDomains.find(d=>d.id==='field-multiscale')!.unitIds.includes(unit.id));
 const content=await getLearningContent(id as LearningUnitId);for(const lang of ['ko','en']as const){assert.equal(content.outline[lang].length,24);assert(!content[lang].includes('katex-error'));assert(!/<h1/.test(content[lang]));assert(content[lang].includes('<math'));}assert(!/[가-힣]/.test(content.en));
});
test('all598 earlier units,guides and51 domain scopes retain exact content and order',()=>{
 assert(learningUnits.length>=599);assert.equal(new Set(learningUnits.map(u=>u.id)).size,learningUnits.length);assert.deepEqual(learningUnits.slice(0,598).map(u=>u.id),before.unitOrder);assert.deepEqual(Object.keys(learningGuides).slice(0,598),before.guideOrder);assert.equal(learningUnits[598].id,id);assert.equal(Object.keys(learningGuides)[598],id);
 for(const unit of learningUnits.slice(0,598)){assert.equal(hash(unit),before.units[unit.id as keyof typeof before.units]);assert.equal(hash(learningGuides[unit.id]),before.guides[unit.id as keyof typeof before.guides]);}
 assert(learningUnits.reduce((n,u)=>n+learningGuides[u.id].experiments.length,0)>=1850);for(const domain of curriculumDomains)assert.equal(hash({...domain,unitIds:domain.unitIds.filter(unitId=>before.unitOrder.includes(unitId))}),before.domains[domain.id as keyof typeof before.domains]);
});
test('reader dispatches euclidean-kernel concepts,cover and embedded lazy experiment without duplicate quizzes',()=>{
 const read=(p:string)=>readFileSync(`src/components/learning/${p}`,'utf8');assert(read('interactive-book.tsx').includes('isEuclideanKernelTeachingId(unit.id) ? <EuclideanKernelReaderLab'));assert(read('concept-walkthrough.tsx').includes('isEuclideanKernelTeachingId(kind)) return <EuclideanKernelConcept'));assert(read('book-art.tsx').includes('isEuclideanKernelTeachingId(kind)) return <EuclideanKernelCover'));assert(read('euclidean-kernel-reader-lab.tsx').includes('id="experiment"'));assert(read('euclidean-kernel-reader-lab.tsx').includes('<Lab language={language} embedded />'));assert(read('euclidean-kernel-lab.tsx').includes('!embedded &&'));
});
test('all eight reference diagrams and catalog cover render with labels and no Worker access',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),savedReact=Object.getOwnPropertyDescriptor(globalThis,'React'),savedWorker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{const{default:Graphic}=await import('../src/components/learning/euclidean-kernel-graphic'),{default:Concept}=await import('../src/components/learning/euclidean-kernel-concept'),{default:Art}=await import('../src/components/learning/book-art'),{LanguageProvider}=createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider');
  for(const language of ['ko','en']as const){for(let step=0;step<8;step++){const svg=renderToStaticMarkup(React.createElement(Graphic,{step,language}));assert(svg.includes('role="img"'));assert(svg.includes('<desc'));assert(svg.includes(`data-euclidean-reference="${step}"`));assert(!svg.includes('NaN'));if(language==='en')assert(!/[가-힣]/.test(svg));}const concept=renderToStaticMarkup(React.createElement(Concept,{language}));assert(concept.includes('href="#experiment"'));}
  const cover=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Art,{kind:id as LearningUnitId})));assert(cover.includes('data-euclidean-reference="0"'));assert(!cover.includes('min-width:720'));assert.equal(calls,0);
 }finally{if(savedReact)Object.defineProperty(globalThis,'React',savedReact);else Reflect.deleteProperty(globalThis,'React');if(savedWorker)Object.defineProperty(globalThis,'Worker',savedWorker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('full reader excludes both field numerical engines and the independent validator',async()=>{
 const result=await build({entryPoints:['src/components/learning/interactive-book.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:root+'/graph',metafile:true,jsx:'automatic',logLevel:'silent'});const inputs=Object.keys(result.metafile!.inputs);assert(!inputs.some(p=>/learning-(?:euclidean-kernel|brownian-bridge|spatial-spectrum|nongaussian-field|gaussian-field|duffing-secular|boundary-layer|quartic-asymptotic|regular-perturbation|graph-gauss|gauge-orbits|field-modes|field-balance|scalar-field|quantum-hilbert-core)(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));assert(inputs.includes('src/components/learning/euclidean-kernel-matrix-engine.ts'));writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(result.metafile,null,2)+'\n');
});
test('every registered bilingual manuscript renders with strict equations and English-only visible copy',async()=>{
 const rows:{id:string;koSections:number;enSections:number}[]=[];
 for(let start=0;start<learningUnits.length;start+=6)await Promise.all(learningUnits.slice(start,start+6).map(async unit=>{const content=await getLearningContent(unit.id);for(const language of ['ko','en']as const){assert(content.outline[language].length>0,unit.id);assert(!content[language].includes('katex-error'),unit.id);assert(!/<h1(?:\s|>)/.test(content[language]),unit.id);}assert(!/[가-힣]/.test(content.en),unit.id);rows.push({id:unit.id,koSections:content.outline.ko.length,enSections:content.outline.en.length});}));
 assert.equal(rows.length,learningUnits.length);writeFileSync(root+'/all-manuscripts.json',JSON.stringify({registeredUnits:learningUnits.length,bilingualManuscripts:rows.length*2,rows},null,2)+'\n');
});
