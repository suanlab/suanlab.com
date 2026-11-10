import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createRequire} from 'node:module';
import {build} from 'esbuild';
import {learningUnits,learningGuides,type LearningUnitId} from '../src/data/interactive-book';
import {curriculumDomains} from '../src/data/learning-curriculum';
import {scalarFieldReader} from '../src/data/scalar-field-reader';
import {scalarFieldTeachingId as id} from '../src/lib/learning-scalar-field-teaching';
import {getLearningContent} from '../src/lib/learning-content';
import before from './fixtures/pre-scalar-field-registry-hashes.json';
const hash=(value:unknown)=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const root='.runtime/learning/scalar-field-reader-integration';mkdirSync(root,{recursive:true});
test('scalar field appends with valid metadata,prerequisites,seven concepts,four guides and five questions',async()=>{
 const unit=learningUnits.find(u=>String(u.id)===id);assert(unit);assert.equal(learningUnits.indexOf(unit),584);assert.deepEqual(unit as unknown,scalarFieldReader.unit);assert.deepEqual(learningGuides[unit.id],scalarFieldReader.guide);assert.equal(unit.questions.length,5);assert.equal(scalarFieldReader.guide.steps.length,7);assert.equal(scalarFieldReader.guide.experiments.length,4);
 assert(unit.prerequisiteIds.every(prior=>learningUnits.findIndex(u=>u.id===prior)>=0&&learningUnits.findIndex(u=>u.id===prior)<584));assert(curriculumDomains.find(d=>d.id==='field-multiscale')!.unitIds.includes(unit.id));
 const content=await getLearningContent(id);for(const lang of ['ko','en']as const){assert.equal(content.outline[lang].length,14);assert(!content[lang].includes('katex-error'));assert(!/<h1/.test(content[lang]));assert(content[lang].includes('<math'));}assert(!/[가-힣]/.test(content.en));
});
test('all584 earlier units,guides and51 domain scopes retain exact content and order',()=>{
 assert(learningUnits.length>=585);assert.equal(new Set(learningUnits.slice(0,585).map(u=>u.id)).size,585);assert.deepEqual(learningUnits.slice(0,584).map(u=>u.id),before.unitOrder);assert.deepEqual(Object.keys(learningGuides).slice(0,584),before.guideOrder);assert.equal(learningUnits[584].id,id);assert.equal(Object.keys(learningGuides)[584],id);
 for(const unit of learningUnits.slice(0,584)){assert.equal(hash(unit),before.units[unit.id as keyof typeof before.units]);assert.equal(hash(learningGuides[unit.id]),before.guides[unit.id as keyof typeof before.guides]);}
 assert.equal(learningUnits.slice(0,585).reduce((n,u)=>n+learningGuides[u.id].experiments.length,0),1765);for(const domain of curriculumDomains)assert.equal(hash({...domain,unitIds:domain.unitIds.filter(unitId=>(before.unitOrder as string[]).includes(unitId))}),before.domains[domain.id as keyof typeof before.domains]);
});
test('reader dispatches scalar concepts,cover and embedded lazy experiment without duplicate quizzes',()=>{
 const read=(p:string)=>readFileSync(`src/components/learning/${p}`,'utf8');assert(read('interactive-book.tsx').includes('isScalarFieldTeachingId(unit.id) ? <ScalarFieldReaderLab'));assert(read('concept-walkthrough.tsx').includes('isScalarFieldTeachingId(kind)) return <ScalarFieldConcept'));assert(read('book-art.tsx').includes('isScalarFieldTeachingId(kind)) return <ScalarFieldCover'));assert(read('scalar-field-reader-lab.tsx').includes('id="experiment"'));assert(read('scalar-field-reader-lab.tsx').includes('<Lab language={language} embedded/>'));assert(read('scalar-field-lab.tsx').includes('!embedded&&'));
});
test('all seven reference diagrams and catalog cover render with labels and no Worker access',async()=>{
 createRequire(`${process.cwd()}/package.json`).extensions['.css']=()=>{};const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),savedReact=Object.getOwnPropertyDescriptor(globalThis,'React'),savedWorker=Object.getOwnPropertyDescriptor(globalThis,'Worker');Object.defineProperty(globalThis,'React',{configurable:true,value:React});let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{const{default:Graphic}=await import('../src/components/learning/scalar-field-graphic'),{default:Concept}=await import('../src/components/learning/scalar-field-concept'),{default:Art}=await import('../src/components/learning/book-art'),{LanguageProvider}=createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider');
  for(const language of ['ko','en']as const){for(let step=0;step<7;step++){const svg=renderToStaticMarkup(React.createElement(Graphic,{step,language}));assert(svg.includes('role="img"'));assert(svg.includes('<desc'));assert(svg.includes(`data-scalar-reference="${step}"`));assert(!svg.includes('NaN'));if(language==='en')assert(!/[가-힣]/.test(svg));}const concept=renderToStaticMarkup(React.createElement(Concept,{language}));assert(concept.includes('href="#experiment"'));}
  const cover=renderToStaticMarkup(React.createElement(LanguageProvider,null,React.createElement(Art,{kind:id as LearningUnitId})));assert(cover.includes('data-scalar-reference="0"'));assert(!cover.includes('min-width:640'));assert.equal(calls,0);
 }finally{if(savedReact)Object.defineProperty(globalThis,'React',savedReact);else Reflect.deleteProperty(globalThis,'React');if(savedWorker)Object.defineProperty(globalThis,'Worker',savedWorker);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('full reader excludes scalar numerical engines and the independent validator',async()=>{
 const result=await build({entryPoints:['src/components/learning/interactive-book.tsx'],bundle:true,write:false,format:'esm',platform:'browser',splitting:true,outdir:root+'/graph',metafile:true,jsx:'automatic',logLevel:'silent'});const inputs=Object.keys(result.metafile!.inputs);assert(!inputs.some(p=>/learning-scalar-field(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));assert(inputs.includes('src/components/learning/scalar-field-plot-engine.ts'));writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(result.metafile,null,2)+'\n');
});
